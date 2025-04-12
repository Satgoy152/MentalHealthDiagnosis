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

class IterativeRagMentalHealthBot:

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def __init__(self):
        print("Starting Bot -----------------------------------###")

        from langsmith import Client
        from retrieval import Retriever
        from dotenv import load_dotenv, get_key
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain.chains import create_history_aware_retriever, create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_community.chat_message_histories import ChatMessageHistory
        from langchain_core.chat_history import BaseChatMessageHistory
        from langchain_core.runnables.history import RunnableWithMessageHistory

        load_dotenv()
        GEMINI_API_KEY = get_key(dotenv_path=".env", key_to_get="GEMINI_API_KEY")

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)
        summary_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)

        with open("retriever_prompt.txt", "r") as f:
            retriever_prompt = f.read()

        with open("prompt.txt", "r") as f:
            prompt = f.read()

        retriever_template = ChatPromptTemplate.from_messages([
            ("system", retriever_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            ("system", "{context}"),
        ])

     
        refined_query_template = ChatPromptTemplate.from_messages([
            ("system", "Refine the user's question to make it clearer and more specific using the following context."),
            ("human", "Original Question: {input}"),
            ("system", "{context}")
        ])

        retriever = Retriever()
        rag_retriver = retriever.retriver_sim

        history_aware_retriever = create_history_aware_retriever(
            summary_llm, rag_retriver, retriever_template
        )

        augmented_chain = create_stuff_documents_chain(llm, prompt_template)
        refined_query_chain = create_stuff_documents_chain(llm, refined_query_template)

        rag_chain = create_retrieval_chain(history_aware_retriever, augmented_chain)

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

        
        self.refined_query_chain = refined_query_chain
        self.retriever = history_aware_retriever
        self.llm = llm

    def iterative_chat_stream(self, text: str):
        session_id = "abc123"
        history = self.store[session_id] if session_id in self.store else ChatMessageHistory()


        initial_docs = self.retriever.invoke({
            "input": text,
            "chat_history": history.messages
        })


        refined_query = self.refined_query_chain.invoke({
            "input": text,
            "context": IterativeRagMentalHealthBot.format_docs(initial_docs)
        })


        final_answer = self.conversational_rag_chain.invoke(
            {"input": refined_query},
            config={"configurable": {"session_id": session_id}}
        )["answer"]

        return final_answer
