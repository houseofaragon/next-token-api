import torch
import tiktoken
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
from fastapi import FastAPI
from pydantic import BaseModel
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

# Pydantic model for input data
class TextInput(BaseModel):
    text: str
    tokenizer_name: str  # New field to select tokenizer

# Function to dynamically load the selected model and tokenizer
def load_model_and_tokenizer(tokenizer_name: str):
    # Load model and tokenizer based on user selection
    if tokenizer_name == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    elif tokenizer_name == 'distilgpt2':
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    elif tokenizer_name == 'EleutherAI/gpt-neo-1.3B':
        model = GPTNeoForCausalLM.from_pretrained(tokenizer_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name == 't5-small':
        model = T5ForConditionalGeneration.from_pretrained(tokenizer_name)
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name == 'facebook/bart-small':
        model = BartForConditionalGeneration.from_pretrained(tokenizer_name)
        tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
    elif tokenizer_name == 'o200k_base':
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = tiktoken.get_encoding('o200k_base')
    else:   
        raise ValueError("Unsupported model name or tokenizer")

    model.eval()
    return model, tokenizer

# Function to get next token probabilities
def get_next_token_probabilities(tokens, model, tokenizer):
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
    tokenizer_name = input_data.tokenizer_name
    
    # Load the selected model and tokenizer
    model, tokenizer = load_model_and_tokenizer(tokenizer_name)
    
    tokens = tokenizer.encode(text)
    print(f"Tokens: {tokens}")
    print(f"Tokens length: {len(tokens)}")
    top_tokens, top_probabilities = get_next_token_probabilities(tokens, model, tokenizer)
    print(model.config.vocab_size)
    return {
        "model_config": model.config,
        "tokens": tokens,
        "top_tokens": top_tokens,
        "top_probabilities": top_probabilities
    }

# Run the API with: uvicorn main:app --reload
