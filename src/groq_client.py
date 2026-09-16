from groq import Groq


MODEL_NAME = "openai/gpt-oss-120b"


class GroqService:
    def __init__(self, api_key: str):
        self.client = Groq(
            api_key=api_key
        )

    def generate_answer(
        self,
        system_prompt,
        user_prompt,
        temperature=0.1,
        max_tokens=3000
    ):
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content
