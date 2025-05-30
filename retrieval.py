from langchain_chroma import Chroma
from langchain_voyageai import VoyageAIEmbeddings
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import chromadb
from dotenv import load_dotenv, get_key
import streamlit as st


# load VoyageAI key
load_dotenv()
        
class Retriever:
    def __init__(self, model: str = "voyage-2") -> None:
        new_client = chromadb.PersistentClient(path = "./chroma_db", tenant = DEFAULT_TENANT, database = DEFAULT_DATABASE, settings = Settings())

        embeddings = VoyageAIEmbeddings(
            voyage_api_key= get_key(dotenv_path = '.env',key_to_get = "VOYAGEAI_KEY"), model="voyage-large-2-instruct")
        
        saved_data_store = Chroma(persist_directory="./chroma_db", collection_name="umich_fa2024", embedding_function=embeddings, client=new_client)

        self.retriver_sim = saved_data_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 2, "score_threshold": 0.3})

