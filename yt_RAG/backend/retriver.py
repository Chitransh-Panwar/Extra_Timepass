from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv


load_dotenv()
api = os.getenv("OPENROUTER_API_KEY")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    api_key=api, 
    base_url="https://openrouter.ai/api/v1"
)

def build_retriver(video_id):
    try:
        transcript_list = YouTubeTranscriptApi().fetch(
            video_id, 
            languages=["en", "pa", "hi", "es", "fr", "de", "pt", "ar", "ru", "ja", "ko", "zh", "it"]
        )
        transcript = " ".join(chunk.text for chunk in transcript_list)
    except TranscriptsDisabled:
        raise RuntimeError("Error: Captions are disabled or unavailable for this YouTube video.")
    except Exception as e:
        raise RuntimeError(f"Error gathering transcript: {str(e)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store