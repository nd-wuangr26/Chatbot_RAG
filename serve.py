from core.embeding.HuggingEmbed import HuggingEmbed
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from langchain_core.documents import Document
from typing import List, TypedDict
from core.chunking.docling_chunk import DoclingChunk
from core.llm.gemini_llm import LLM
from config.config import *
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# document_path = DOCUMENT_PATH
api_key = os.getenv("Gemini_api_key")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Load or create vector store
vector_Hugging = HuggingEmbed()
try :
    vector_store = vector_Hugging.load_vector_store()
    print("Vector store loaded successfully")
except FileNotFoundError:
    print("Vector store not found, creating a new one")
    processor = DoclingChunk()
    chunks = processor.load_docling()
    print(chunks[:1])
    # documents = processor.load_pdf(document_path)
    # chunks = processor.split_text(documents)
    print("Text chunks created successfully")

    vector_store = vector_Hugging.create_vector_store(chunks)
    vector_Hugging.save_vector_store()
    print("Vector store created and saved successfully")


def retrivel(state: State) -> State:
    similarity_search = vector_store.similarity_search(query=state['question'], k=5)
    return {**state,
        "context": similarity_search}


def generate(state: State):
    llm = LLM(api_key=api_key)
    context_text = "\n".join([doc.page_content for doc in state['context']])
    prompt = f"Ban la tro ly tuyen sinh da nganh nghe. Cau hoi cua nguoi dung la: {state['question']} Tra loi cau hoi dua theo noi dung cua doan nay\n{context_text}"
    print("Generated prompt:", prompt)
    answer = llm.post_request(prompt)
    return {
        **state,
        "answer": answer
    }

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Có thể thay đổi thành domain cụ thể cho production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(request: QuestionRequest) -> Dict[str, Any]:
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    # Tìm kiếm context liên quan
    state = {"question": request.question, "context": [], "answer": ""}
    state = retrivel(state)
    # Sinh đáp án dựa trên context
    final_state = generate(state)
    return final_state


@app.get("/")
async def root():
    return {"message": "Welcome to the RAG service!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8001, reload=True)