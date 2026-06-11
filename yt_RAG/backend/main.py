from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from yt_rag import ask_youtube

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"]
)

class QueryPayload(BaseModel):
    url:str
    video_id:Optional[str] = None
    question:str

@app.post("/ask")
async def ask_endpoint(payload:QueryPayload):
    if payload.video_id is not None:
        result=ask_youtube(payload.video_id,payload.question)
        return {"answer":result} 
    return {"answer":"web scraping coming soon"}




