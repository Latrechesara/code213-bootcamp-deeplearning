from huggingface_hub import InferenceClient
from src.prompt import system_instruction

# my token never expose it drirectly for security put it in .env file 

HF_TOKEN = "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"

# Initialize client 
client= InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HF_TOKEN
)
# keep connversation history 
messages= [
    {"role": "system","content": system_instruction}
]

def ask_order(messages):
    """
    Send the conversation messages to Hugging Face chat model and return the respons
    """
    response = client.chat_completion(messages=messages)
    return response.choices[0].message['content']
