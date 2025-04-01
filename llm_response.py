__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
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




class MentalHealthBot:

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def __init__(self):

        print("Starting Bot -----------------------------------###")

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

        client = Client()

        print("Initializing RAG system")
        retriever = Retriever()

        # create retrievers for audit(dummy) and chat(rag)
        rag_retriver = retriever.retriver_sim
        dummy_retriever = retriever.retriever_dummy

        print("Initializing LLM")
        # llm = ChatOpenAI(temperature=0.7, model= "gpt-4o-mini-2024-07-18", api_key=st.secrets["OPENAI_KEY"], streaming=True)
        # summary_llm = ChatAnthropic(temperature=0.7, model="claude-3-5-sonnet-20240620", api_key=st.secrets["ANTHROPIC_KEY"])
        # dummy_llm = ChatOpenAI(temperature=0.7, model= "gpt-4o-mini-2024-07-18", api_key=st.secrets["OPENAI_KEY"], max_tokens=1)

        GEMINI_API_KEY = "AIzaSyC_e5VPxiDbXjG09f3ZLBdvt6XEJU9lmRY"
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
        response = model.generate_content("Write a poem about AI.")
        print(response.text)
        with open("retriever_prompt.txt", "r") as f:
            retriever_prompt = f.read()


        with open("tara_prompt.txt", "r") as f:
            tara_prompt = f.read()
        
        retriever_template = ChatPromptTemplate.from_messages(
            [
                ("system", retriever_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        tara_prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", tara_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        # audit_summary_template = ChatPromptTemplate.from_template(audit_summary_prompt)

        history_aware_retriever = create_history_aware_retriever(
            summary_llm, rag_retriver, retriever_template
        )


        print("Creating RAG chain")
        
        #create chain to insert documents for context (rag documents)
        augmented_chain = create_stuff_documents_chain(llm, tara_prompt_template)

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
        print(self.store["abc123"])
        return response