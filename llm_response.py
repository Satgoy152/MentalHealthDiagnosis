import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import asyncio
from typing import AsyncGenerator
from langsmith import Client 
import streamlit as st
from retrieval import Retriever
import google.generativeai as genai
from dotenv import load_dotenv, get_key




class MentalHealthBot:

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def __init__(self):

        print("Starting Bot -----------------------------------###")

        # os.environ["LANGCHAIN_TRACING_V2"] = "true"
        # os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        # os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

        client = Client()

        print("Initializing RAG system")
        retriever = Retriever()

        # create retrievers for audit(dummy) and chat(rag)
        rag_retriver = retriever.retriver_sim
        # dummy_retriever = retriever.retriever_dummy

        print("Initializing LLM")
        
        load_dotenv()
        GEMINI_API_KEY = get_key(dotenv_path=".env", key_to_get="GEMINI_API_KEY")

        # genai.configure(api_key=GEMINI_API_KEY)

        # use LangChain's wrapper for Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)
        summary_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)
        # response = gemini_llm.generate_content("Write a poem about AI.")
        # print(response.text)

        with open("retriever_prompt.txt", "r") as f:
            retriever_prompt = f.read()


        with open("prompt.txt", "r") as f:
            prompt = f.read()
        
        retriever_template = ChatPromptTemplate.from_messages(
            [
                ("system", retriever_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                ("system", "{context}"),
            ]
        )
        # audit_summary_template = ChatPromptTemplate.from_template(audit_summary_prompt)

        history_aware_retriever = create_history_aware_retriever(
            summary_llm, rag_retriver, retriever_template
        )


        print("Creating RAG chain")
        
        #create chain to insert documents for context (rag documents)
        augmented_chain = create_stuff_documents_chain(llm, prompt_template)

        # chain that retrieves documents and then passes them to the question_answer_chain

        rag_chain = create_retrieval_chain(history_aware_retriever, augmented_chain)
        # audit_text_chain = create_retrieval_chain(audit_retrevier, tara_chain)

        self.store = {}

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self.store:
                print("Creating new chat history for session_id", session_id)
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]


        self.conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    
    def chat_stream(self, text: str):
        response = self.conversational_rag_chain.invoke(
            {"input": text},
                config={
                    "configurable": {"session_id": "abc123"}
                },  # constructs a key "abc123" in `store`.
            )["answer"]
        # print(self.store["abc123"])
        return response