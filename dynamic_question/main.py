import asyncio
from dynamic_question import is_questionable_prompt, generate_follow_up_question

async def main():
    user_prompt = input("Enter a user prompt: ")
    
    # Check if the prompt is questionable
    if await is_questionable_prompt(user_prompt):
        print("The prompt is questionable.")
        follow_up_question = await generate_follow_up_question(user_prompt)
        print(f"Follow-up Question: {follow_up_question}")
    else:
        print("The prompt is clear and self-contained. No follow-up question needed.")

if __name__ == "__main__":
    asyncio.run(main())
