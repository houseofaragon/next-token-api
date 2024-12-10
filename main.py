import tiktoken
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware


# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can specify your frontend domain here)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Pydantic model for input data
class TextInput(BaseModel):
    text: str

# Function to get next token probabilities
def get_next_token_probabilities(tokens):
    input_ids = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]  # Get logits for the last token

    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_probabilities, top_indices = torch.topk(probs, 10)
    
    top_tokens = [tokenizer.decode([idx]) for idx in top_indices[0]]
    top_probabilities = top_probabilities[0].tolist()
    
    return top_tokens, top_probabilities

# Define the API endpoint
@app.post("/predict")
async def predict_next_token(input_data: TextInput):
    print(input_data)
    text = input_data.text
    tokens = tokenizer.encode(text)
    top_tokens, top_probabilities = get_next_token_probabilities(tokens)
    
    return {
        "top_tokens": top_tokens,
        "top_probabilities": top_probabilities
    }

# Run the API with: uvicorn main:app --reload
"""
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "The boy jumped over the"
}'
"""