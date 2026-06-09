from dotenv import load_dotenv
import os
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser



load_dotenv()
api=os.getenv("OPENROUTER_API_KEY")

llm=ChatOpenAI(
    model="openrouter/owl-alpha",
    api_key=api,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.5
)



video_id="EHLI2WZUtXs" 



def format_docs(retriever_docs):
    context_text="\n\n".join(doc.page_content for doc in retriever_docs)
    return context_text

try:
    transcript_list=YouTubeTranscriptApi().fetch(video_id,languages=["en"])
    
    transcript=" ".join(chunk.text for chunk in transcript_list )
    
except TranscriptsDisabled:
    print("NO captions available for this video ")

splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunk=splitter.create_documents([transcript])
embeddings=OpenAIEmbeddings(model="text-embedding-3-small",api_key=api,base_url="https://openrouter.ai/api/v1")
vector_store=FAISS.from_documents(chunk,embeddings)
vector_list = list(vector_store.index_to_docstore_id.values())

retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":1})

prompt=PromptTemplate(
    template="""
    you are a helpful assistant.
    Answer only from the provided transcript context.if the context is insufficient ,just say you dont know .
    {context}
    question:{question}
    
    """,
    input_variables=['context','question']
)

question="is the topic of openai most powerful model discussed in this video ? if yes then what was discussed."
retriever_docs=retriever.invoke(question)
parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'context':retriever | RunnableLambda(format_docs),
    'question':RunnablePassthrough()
})

main_chain=parallel_chain | prompt | llm |parser

print(main_chain.invoke('when did roman empire form  and who was the main person responsible for its creation ?'))


