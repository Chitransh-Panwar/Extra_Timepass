import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    model="openrouter/owl-alpha",
    api_key=api,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.5
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    api_key=api, 
    base_url="https://openrouter.ai/api/v1"
)

def format_docs(retriever_docs):
    return "\n\n".join(doc.page_content for doc in retriever_docs)

def ask_youtube(video_id: str, question: str) -> str:
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        transcript = " ".join(chunk.text for chunk in transcript_list)
        
    except TranscriptsDisabled:
        return "Error: Captions are disabled or unavailable for this YouTube video."
    except Exception as e:
        return f"Error gathering transcript: {str(e)}"

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    prompt = PromptTemplate(
        template="""
        you are a helpful assistant.
        Answer only from the provided transcript context.if the context is insufficient ,just say you dont know .
        {context}
        question:{question}
        """,
        input_variables=['context', 'question']
    )

    parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | llm | parser

    try:
        response_text = main_chain.invoke(question)
        return response_text
    except Exception as e:
        return f"Error executing pipeline generation: {str(e)}"
