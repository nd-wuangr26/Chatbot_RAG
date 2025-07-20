from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from core.embeding.base import BaseEmbedding
from config.config import *

import os

class HuggingEmbed(BaseEmbedding):
    def __init__(self, name: str = MODEL_NAME_EMBEDDING):
        self.embeddings = HuggingFaceEmbeddings(model_name=name)
        self.vector_db = None

    def create_vector_store(self, documents: Document) -> FAISS:
        """Create vector store from documents."""
        self.vector_db = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        return self.vector_db
    
    def save_vector_store(self, path: str = "vectordb") -> None:
        """Save vector store to specified path."""
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            if self.vector_db is not None:
                self.vector_db.save_local(path)
        else:
            raise FileExistsError(f"Vector store already exists at {path}")
        
    def load_vector_store(self, path: str = "vectordb") -> None:
        """Load vector store from specified path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector store not found at {path}")
        self.vector_db = FAISS.load_local(
            path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        return self.vector_db
    
    