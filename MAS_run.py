import os
import json
import logging
import requests
import re
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

# Standardized Prompts (Imported from your prompts.py)
# from prompts import (
#     INSTRUCTION_PLANNER, INSTRUCTION_ALLOCATOR, 
#     INSTRUCTION_OBSERVER, INSTRUCTION_EXECUTOR, 
#     MOBILE_ACTION_SPACE, Web_PC_ACTION_SPACE
# )

# Logging Configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LAMO-Agent")

# --- Supplementary Prompts ---

ACT2SUM_PROMPT = """Summarize the current GUI action in one sentence.
Goal: <goal>{GOAL}</goal>
Current action: <action>{ACTION}</action>
Output Format: <summary>The summary here</summary>"""

HISTORY_SUMMARIZER_PROMPT = """Analyze the historical interaction log and current screen state.
**Interaction History**
<history>{HISTORY}</history>
**Current execution**
<action>{ACTION}</action>

- Note: Monitor if the goal is achieved to avoid repetition.
Output Format: <summary>Detailed history summary</summary>"""

# --- Utility Classes ---

class ResponseParser:
    """Utility to safely extract content from XML-style tags."""
    @staticmethod
    def extract_tag(text: str, tag: str, default: str = "") -> str:
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip() if text else default

class LLMClient:
    """Client for communicating with the VLM (Qwen2.5-VL-3B-Instruct)."""
    def __init__(self, api_url: str = "http://localhost:8000/v1/chat/completions"):
        self.api_url = api_url
        self.headers = {"Content-Type": "application/json"}

    def generate(self, img_url: str, prompt: str, temperature: float = 0.0) -> str:
        """Sends a multimodal request to the VLM API."""
        payload = {
            "model": "Qwen2.5-VL-3B-Instruct",
            "messages": [
                {"role": "system", "content": "You are a specialized GUI automation assistant."},
                {
                    "role": "user", 
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_url}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "temperature": temperature
        }
        
        try:
            response = requests.post(self.api_url, json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            logger.error(f"API Request failed: {e}")
            return ""
        except (KeyError, IndexError) as e:
            logger.error(f"Malformed API response: {e}")
            return ""

# --- Core Agent System ---

class LAMOAgent:
    """Multi-Agent System for GUI Automation."""
    def __init__(self, client: LLMClient):
        self.client = client
        self.parser = ResponseParser()

    def observe(self, img_url: str) -> str:
        """Role: Observer - Converts visual state to text description."""
        from prompts import INSTRUCTION_OBSERVER
        res = self.client.generate(img_url, INSTRUCTION_OBSERVER, temperature=0.2)
        return self.parser.extract_tag(res, "screen2word")

    def plan(self, goal: str, img_url: str) -> Dict[str, Any]:
        """Role: Planner - Breaks down the goal into logic steps."""
        from prompts import INSTRUCTION_PLANNER
        prompt = INSTRUCTION_PLANNER.format(GOAL=goal)
        res = self.client.generate(img_url, prompt, temperature=0.3)
      
        # Robust JSON extraction
        try:
            json_str = self.parser.extract_tag(res, "json")
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Failed to decode Planner JSON. Returning raw string.")
            return {"Planning": res, "Tips": "No specific tips provided."}

    def allocate(self, screen_desc: str, plan: str, history: str, tips: str, img_url: str) -> str:
        """Role: Allocator - Decides the next high-level action."""
        from prompts import INSTRUCTION_ALLOCATOR
        prompt = INSTRUCTION_ALLOCATOR.format(
            SCREEN_DESCRIPTION=screen_desc,
            PLAN=plan,
            HISTORY=history,
            TIPS=tips
        )
        res = self.client.generate(img_url, prompt, temperature=0.2)
        return self.parser.extract_tag(res, "action")

    def execute_mapping(self, action: str, img_url: str) -> str:
        """Role: Executor - Maps description to specific tool code/command."""
        from prompts import INSTRUCTION_EXECUTOR, MOBILE_ACTION_SPACE
        prompt = INSTRUCTION_EXECUTOR.format(
            ACTION_SPACE=MOBILE_ACTION_SPACE,
            ACTION=action
        )
        res = self.client.generate(img_url, prompt, temperature=0.05)
        return res # Usually returns the raw formatted tool command

# --- Main Workflow ---

def run_automation_task(goal: str, env: Any):
    """Main execution loop for GUI navigation."""
    client = LLMClient()
    agent = LAMOAgent(client)
    
    max_steps = 20
    history_window_size = 10
    interaction_history = []

    logger.info(f"Starting Task: {goal}")

    for step in range(1, max_steps + 1):
        logger.info(f"Step {step}/{max_steps}")
        
        try:
            # 1. Environment State
            screen_url = env.get_screen()
            
            # 2. Format History
            if not interaction_history:
                current_history_text = "Task initial state."
            else:
                recent_history = interaction_history[-history_window_size:]
                current_history_text = "\n".join([f"Step {i+1}: {act}" for i, act in enumerate(recent_history)])

            # 3. MAS Workflow
            # Observer
            screen_desc = agent.observe(screen_url)
            logger.info(f"Observation: {screen_desc[:100]}...")

            # Planner
            planning_data = agent.plan(goal, screen_url)
            plan = planning_data.get("Planning", "Continue task.")
            tips = planning_data.get("Tips", "")

            # Allocator
            next_action = agent.allocate(screen_desc, plan, current_history_text, tips, screen_url)
            logger.info(f"Planned Action: {next_action}")

            # Executor
            tool_call = agent.execute_mapping(next_action, screen_url)
            logger.info(f"Tool Command: {tool_call}")

            # 4. Environment Execution
            if not tool_call:
                logger.error("No valid tool call generated. Aborting.")
                break

            env.execution(tool_call)
            interaction_history.append(next_action)

            # Optional: Add judge logic to check if goal is achieved
            if env.is_goal_achieved(goal):
                logger.info("Goal achieved successfully.")
                break

        except Exception as e:
            logger.critical(f"Unexpected error at step {step}: {e}")
            break

    env.final_judge(goal)

if __name__ == "__main__":
    # Example Initialization
    # env = YourEnvironment(goal)
    # run_automation_task("Open YouTube and search for Lo-fi music", env)
    pass
