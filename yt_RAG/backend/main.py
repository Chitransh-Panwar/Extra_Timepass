import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from yt_rag import YouTubeRAGBot
from scraper import ask_webpage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"]
)

active_sessions = {}

class QueryPayload(BaseModel):
    url: str
    video_id: Optional[str] = None
    question: str
    session_id: Optional[str] = None

@app.post("/ask")
async def ask_endpoint(payload: QueryPayload):
    if payload.video_id is not None:
        session_key = payload.session_id if payload.session_id else f"{payload.video_id}_{uuid.uuid4().hex[:8]}"
        
        try:
            if session_key not in active_sessions:
                active_sessions[session_key] = YouTubeRAGBot(video_id=payload.video_id)
            else:
                bot = active_sessions[session_key]
                if bot.video_id != payload.video_id:
                    bot.update_video(payload.video_id)
            
            bot = active_sessions[session_key]
            result = bot.ask(payload.question)
            
            return {
                "answer": result,
                "session_id": session_key
            }
            
        except Exception as e:
            return {"error": f"Failed to process request: {str(e)}"}
    
    result=await ask_webpage(payload.url, payload.question)
    return {"answer":result}
