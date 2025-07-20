from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class VectorStore():
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_db = None

    def create_vector_store(self, chunks: Document) -> FAISS:
        self.vector_db = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        return self.vector_db
    
    def save_vector_store(self, path: str) -> None:
        """
        Save the vector store to the specified path.
        """
        if self.vector_db is not None:
            self.vector_db.save_local(path)

    def load_vector_store(self, path: str) -> None:
        """
        Load the vector store from the specified path.
        """
        self.vector_db = FAISS.load_local(
            path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )

# if __name__ == "__main__":
    # embedding = Embedding()
    # processor = ProcessData()
    # pdf_path = Path("/home/quang/Downloads/Ky thuat xay dung-1718.pdf")
    # documents = processor.load_pdf(pdf_path)
    # chunks = processor.split_text(documents)
    # texts = [chunks[:1], chunks[:2], chunks[3:4]]  # Giả sử bạn có một danh sách các đoạn văn bản
    # for text in texts:
    #     vector_store = embedding.create_vector_store(text)
    #     embedding.save_vector_store("vector_store_path")
    #     print(vector_store)
    # embedding.load_vector_store("vector_store_path")
    # print("Vector store loaded successfully")
    

   