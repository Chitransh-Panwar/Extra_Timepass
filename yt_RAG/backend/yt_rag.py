import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from retriver import build_retriver

load_dotenv()
api = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    model="openrouter/owl-alpha",
    api_key=api,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.5
)

class YouTubeRAGBot:
    def __init__(self, video_id: str):
        self.video_id = video_id
        self.chat_history_store = InMemoryChatMessageHistory()
        self.retriever = self._build_retriever()
        self.contextual_chain = self._build_contextual_chain()
        self.main_chain = self._build_main_chain()

    def _build_retriever(self):
        vector_store = build_retriver(self.video_id)
        return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    def update_video(self, new_video_id: str):
        self.video_id = new_video_id
        self.retriever = self._build_retriever()
        self.main_chain = self._build_main_chain()

    def _build_contextual_chain(self):
        contextualize_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
        )
        contextual_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        return contextual_prompt | llm | StrOutputParser()

    def _build_main_chain(self):
        def format_docs(retriever_docs):
            return "\n\n".join(doc.page_content for doc in retriever_docs)

        prompt = PromptTemplate(
            template="""You are a helpful assistant analyzing a YouTube video transcript.
            Use the provided context to answer the question as best you can.
            If the context has partial information, use it along with your knowledge to give a helpful answer.
            If there is truly no relevant information, say so briefly.

            Context:
            {context}

            Question: {question}""",
            input_variables=['context', 'question']
        )

        parallel_chain = RunnableParallel({
            'context': lambda x: format_docs(self.retriever.invoke(x)),
            'question': RunnablePassthrough()
        })
        
        return parallel_chain | prompt | llm | StrOutputParser()

    def ask(self, question: str) -> str:
        try:
            history_messages = self.chat_history_store.messages
            
            standalone_question = self.contextual_chain.invoke({
                "chat_history": history_messages,
                "input": question
            })

            response_text = self.main_chain.invoke(standalone_question)
            
            self.chat_history_store.add_user_message(question)
            self.chat_history_store.add_ai_message(response_text)
            
            return response_text
        except Exception as e:
            return f"Error executing pipeline generation: {str(e)}"
