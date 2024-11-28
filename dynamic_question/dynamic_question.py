import os
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants for OpenRouter API
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

async def call_qwen(prompt: str) -> str:
    """
    Asynchronously call Qwen 2 7B Instruct API via OpenRouter for responses.
    
    :param prompt: The prompt to be sent to the Qwen model.
    :return: The response text generated by the Qwen model.
    """
    payload = {
        "model": "qwen/qwen-2-7b-instruct:free",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(MODEL_URL, headers=HEADERS, json=payload)
            
            # Log response status and text for debugging
            print(f"Response Status Code: {response.status_code}")
            print(f"Response Text: {response.text}")
            
            if response.status_code == 200:
                response_json = response.json()
                # Check if response has expected fields
                if "choices" in response_json and response_json["choices"]:
                    return response_json["choices"][0]["message"]["content"]
                else:
                    raise ValueError("Unexpected API response format.")
            else:
                raise Exception(f"API Error: {response.status_code} - {response.text}")

        except httpx.RequestError as e:
            print(f"Request Error: {e}")
            raise
        except ValueError as e:
            print(f"Value Error: {e}")
            raise

async def is_questionable_prompt(prompt: str) -> bool:
    """
    Check if the prompt is questionable.
    """
    system_prompt = f"""
    Determine if the following prompt requires clarification. 

    A prompt requires clarification if:
    1. It lacks critical details necessary to provide an accurate and complete response.
    2. It asks for a process, setup, or solution that depends on variables like operating system, software version, or hardware configuration, but these variables are not specified.
    3. It is ambiguous or could lead to multiple interpretations.

    Examples of prompts requiring clarification:
    - "How to install AWS CLI?"
    - "Set up Python."
    - "What are the prerequisites for Docker?"
    - "Install software on my system."

    Examples of clear and self-contained prompts:
    - "How to install AWS CLI on macOS?"
    - "Install Python 3.9 on Ubuntu 20.04."
    - "What are the benefits of Kubernetes?"
    - "How to set up Docker on a Windows 10 system?"

    Respond only with:
    - "True" if the prompt requires clarification.
    - "False" if the prompt is clear, self-contained, and requires no clarification.

    Prompt: {prompt}
    """

    response = await call_qwen(system_prompt)
    
    # Log and interpret response
    print(f"is_questionable_prompt Response: {response}")
    return response.strip().lower() == "true"  # Ensure case-insensitivity

async def generate_follow_up_question(prompt: str) -> str:
    """
    Generate a follow-up question to clarify the user's input.

    :param prompt: The user's input prompt.
    :return: A follow-up question generated by Qwen API.
    """
    follow_up_prompt = f"""
    Generate a single concise follow-up question to clarify the following prompt:

    {prompt}

    The question should directly address any missing critical details or variables needed to provide a clear and complete answer. Only return the question without any additional context.
    """

    response = await call_qwen(follow_up_prompt)
    print(f"generate_follow_up_question Response: {response}")
    return response