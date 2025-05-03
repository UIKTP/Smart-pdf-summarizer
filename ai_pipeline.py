from attr.validators import max_len
from dotenv import load_dotenv
import os
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import warnings
from langchain_core._api import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

if not HUGGINGFACE_API_KEY:
    raise RuntimeError("Missing HUGGINGFACE_API_KEY in environment.")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_KEY

# Settings
CHROMA_DIR = "./chroma_db"
CHROMA_COLLECTION = "pdf_docs"
SUMMARY_CHUNKS = 10  # Number of chunks to use for summarization
SIMILARITY_THRESHOLD = 0.85

# Initialize tools
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_and_split_pdf(path: str) -> List[Document]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No PDF found at {path}")
    loader = PyPDFLoader(path)
    pages = loader.load()
    return text_splitter.split_documents(pages)

def init_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION
    )

def ingest_documents(docs: List[Document]) -> None:
    db = init_vectorstore()
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    db.add_texts(texts=texts, metadatas=metadatas)
    db.persist()
    print(f"Ingested {len(docs)} chunks into ChromaDB.")

def filter_by_similarity(query: str, vectorstore: Chroma, threshold: float = SIMILARITY_THRESHOLD, k: int = 3) -> List[Document]:
    try:
        scored_docs: List[Tuple[Document, float]] = vectorstore.similarity_search_with_score(query, k=k)
        relevant_docs = [doc for doc, score in scored_docs if score >= threshold]
        return relevant_docs
    except Exception as e:
        print(f"Error during similarity filtering: {e}")
        return []

def run_qa(query: str) -> str:
    vectorstore = init_vectorstore()
    relevant_docs = filter_by_similarity(query, vectorstore)
    if not relevant_docs:
        return "No relevant information found in the PDF."

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = HuggingFaceHub(
        task="text-generation",
        repo_id="google/flan-t5-large",
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        model_kwargs={"temperature": 0.4, "max_length": 1024}
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="refine",
        retriever=retriever
    )
    return qa.run(query)

def summarize_documents(docs: List[Document], length: str = "medium") -> str:
    """
    Summarize a list of Document chunks via a chat‐style prompt.
    - length: "short" (2–3 sentences), "medium" (1 paragraph), "long" (7–10 sentences)
    """
    length_map = {
        "short": "Please summarize the following text in 2–3 sentences:",
        "medium": "Please summarize the following text in a paragraph (4–6 sentences):",
        "long": "Please provide a comprehensive summary of the following text in 7–10 sentences:"
    }
    instruction = length_map.get(length, length_map["medium"])
    content = "\n\n".join([doc.page_content for doc in docs[:SUMMARY_CHUNKS]])
    messages = [
        SystemMessage(content=instruction),
        HumanMessage(content=content)
    ]

    llm = HuggingFaceHub(
        repo_id="facebook/bart-large-cnn",
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        model_kwargs={"temperature": 0.6}
    )
    return llm.invoke(messages)

def main():
    pdf_path = os.path.join("uploads", "sample.pdf")

    print("Loading and splitting PDF…")
    docs = load_and_split_pdf(pdf_path)

    print("Ingesting into vectorstore…")
    ingest_documents(docs)

    print("\nGenerating document summary…")
    summary = summarize_documents(docs, length="medium")
    print("\n=== SUMMARY ===")
    print(summary)

    question = "What is the purpose of the AI pipeline?"
    print(f"\nAsking: {question}")
    answer = run_qa(question)
    print("\n=== ANSWER ===")
    print(answer)

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
