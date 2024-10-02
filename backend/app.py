from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY not found in environment variables.")
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI(title="AI-Powered Teaching Assistant")

origins = [
    "http://localhost:3000", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            
    allow_credentials=True,
    allow_methods=["*"],             
    allow_headers=["*"],             
)

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str

def format_response(response_text: str) -> str:
    formatted_text = response_text.replace("**", "").replace("*", "")   
    return formatted_text

try:
    models = genai.list_models()
    selected_model = None
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            selected_model = m.name
            logger.info(f"Selected model: {selected_model}")
            break
    if not selected_model:
        logger.error("No suitable model found that supports 'generateContent'.")
        raise ValueError("No suitable model found that supports 'generateContent'.")

    model = genai.GenerativeModel(selected_model)
    chat_session = model.start_chat(history=[])
except Exception as e:
    logger.error(f"Error initializing generative model: {e}")
    raise e

@app.post("/chat/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_input = request.prompt.strip()
    
    if not user_input:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    
    try:
        response = chat_session.send_message(user_input, stream=True)
        
        response_text = ""
        for chunk in response:
            if chunk.text:
                response_text += chunk.text
        
        formatted_response = format_response(response_text)
        
        logger.info(f"User: {user_input}")
        logger.info(f"Bot: {formatted_response}")
        
        return ChatResponse(response=formatted_response)
    except Exception as e:
        logger.error(f"Error during chat: {e}")
        raise HTTPException(status_code=500, detail="I'm unable to provide an answer right now. Please try again later.")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the AI-Powered Teaching Assistant!"}
