import os
import requests
from dotenv import load_dotenv
from prompt import (
    Web_PC_ACTION_SPACE,
    INSTRUCTION_EXECUTOR,
    MOBILE_ACTION_SPACE,
    HIGH_LEVEL_PLANNER_INSTRUCTION,
    GPT5_PLANNER_ANDROID_WORLD
)

# Load environment variables
load_dotenv()

# API Configuration
PLANNER_BASE_URL = os.getenv("PLANNER_BASE_URL", "")
PLANNER_API_KEY = os.getenv("PLANNER_API_KEY", "")
LAMO_BASE_URL = os.getenv("LAMO_BASE_URL", "")
LAMO_API_KEY = os.getenv("LAMO_API_KEY", "")

class Agent:
    """
    Agent class responsible for generating actions.
    """

    def __init__(self, device_type: str, planner_model: str = "gpt-5", executor_model: str = "LAMO-3B"):
        """
        Initialize the Agent.

        Args:
            device_type (str): The type of device ('mobile' or 'computer').
            planner_model (str): The name of the model used for planning.
            executor_model (str): The name of the model used for execution.
        """
        self.history = []
        self.notes = []
        self.planner_model = planner_model
        self.executor_model = executor_model
        self.device_type = device_type

    def chat_planner(self, goal: str, b64_img_url: str) -> str:
        """
        Send the current state and goal to the planner model to get the next high-level action.

        Args:
            goal (str): The user's ultimate goal.
            b64_img_url (str): The base64 encoded image URL of the current screen.

        Returns:
            str: The response from the planner model containing the thought, note, and action.
        """
        url = PLANNER_BASE_URL.rstrip("/") + "/chat/completions"
        
        # Use the specific system prompt for the planner (currently defaults to GPT-5 in Android World). 
        system_prompt = GPT5_PLANNER_ANDROID_WORLD 

        # Construct history string from the last 10 steps
        hist_str = ""
        hist_start_idx = max(0, len(self.history) - 10)
        for i, h in enumerate(self.history[-10:], start=hist_start_idx):
            hist_str += f"Step {i + 1}: {h}\n"
        
        # Construct notes string
        notes_str = ""
        for i, n in enumerate(self.notes):
            notes_str += f"(Note {i + 1}): {n}\n"
        
        # Format the user prompt with history, notes, and goal
        prompt = HIGH_LEVEL_PLANNER_INSTRUCTION.format(
            HISTORY=hist_str,
            NOTES=notes_str,
            GOAL=goal,
        )

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt},
                ]
            },
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": b64_img_url}}
                ]
            }
        ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {PLANNER_API_KEY}"
        }
        
        payload = {
            "model": self.planner_model,
            "messages": messages,
            "stream": False,
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
        )
        
        return response.json()["choices"][0]["message"]["content"]

    def chat_lamo(self, action: str, b64_img_url: str) -> str:
        """
        Send the high-level instrcution to the executor model to get specific tool calls.

        Args:
            action (str): The high-level action description from the planner.
            b64_img_url (str): The base64 encoded image URL of the current screen.

        Returns:
            str: The response from the executor model containing the tool call.
        """
        url = LAMO_BASE_URL.rstrip("/") + "/chat/completions"
        
        # Select the appropriate action space based on device type
        action_space = MOBILE_ACTION_SPACE if self.device_type == "mobile" else Web_PC_ACTION_SPACE
        LAMO_EXECUTOR_INSTRUCTION = INSTRUCTION_EXECUTOR
        prompt = LAMO_EXECUTOR_INSTRUCTION.format(
            ACTION=action,
            ACTION_SPACE=action_space,
        )

        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": b64_img_url}}
                ]
            }
        ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LAMO_API_KEY}"
        }

        payload = {
            "model": self.executor_model,
            "messages": messages,
            "stream": False,
            "temperature": 0.05,
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
        )
        
        return response.json()["choices"][0]["message"]["content"]
    
    def chat(self, goal: str, b64_img_url: str) -> str:
        """
        Main chat loop: Plans the next action and then executes it.

        Args:
            goal (str): The user's ultimate goal.
            b64_img_url (str): The base64 encoded image URL of the current screen.

        Returns:
            str: The final tool call string to be executed.
        """
        # Get the high-level plan from the planner
        planner_response = self.chat_planner(goal, b64_img_url)
        
        # Parse the planner response for notes and actions
        note = planner_response.split("<note>")[-1].split("</note>")[0].strip()
        action = planner_response.split("<action>")[-1].split("</action>")[0].strip()

        # Update history and notes
        self.history.append(action)
        if note:
            self.notes.append(note)
        
        # Get the low-level execution command from LAMO
        executor_response = self.chat_lamo(action, b64_img_url)
        
        # Parse the executor response for the tool call
        final_action = executor_response.split("<tool_call>")[-1].split("</tool_call>")[0].strip()

        return final_action

    def reset(self):
        """
        Reset the agent's history and notes to start a new task.
        """
        self.history = []
        self.notes = []
    
def get_screenshot() -> str:
    raise NotImplementedError("This function should return a base64 encoded screenshot of current device screen.")

def exectute_action(action: str) -> bool:
    raise NotImplementedError("This function should execute the given action on the device and return whether the task is finished.")

def main():
    """
        example usage of the Agent class.
    """
    # Initialize the environment and connect the device.
    #Please refer to: https://github.com/xlang-ai/OSWorld and https://github.com/google-research/android_world
    agent = Agent(device_type="mobile", planner_model="gpt-5", executor_model="LAMO-3B")
    goal = "Book a flight to New York for next Monday."
    task_finished = False
    while not task_finished:
        screenshot = get_screenshot()
        agent_action = agent.chat(goal, screenshot)
        task_finished = exectute_action(agent_action)
    agent.reset()
    
if __name__ == "__main__":
    main()

    
    
