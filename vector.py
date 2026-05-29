from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent

POLICY_FOLDER = BASE_DIR / "policies"
DB_LOCATION = BASE_DIR / "chroma_db"
COLLECTION_NAME = "hr_policy_collection"


def get_embedding_function():
    return OllamaEmbeddings(model="mxbai-embed-large:latest")


def load_pdf_documents():
    if not POLICY_FOLDER.exists():
        raise FileNotFoundError(f"Folder not found: {POLICY_FOLDER}")

    pdf_files = list(POLICY_FOLDER.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found inside: {POLICY_FOLDER}")

    print("PDF files found:")
    for pdf in pdf_files:
        print("-", pdf.name)

    loader = PyPDFDirectoryLoader(str(POLICY_FOLDER))
    documents = loader.load()

    documents = [
        doc for doc in documents
        if doc.page_content and doc.page_content.strip()
    ]

    print(f"Pages with readable text: {len(documents)}")

    if not documents:
        raise ValueError("No readable text found in the PDF.")

    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)

    chunks = [
        chunk for chunk in chunks
        if chunk.page_content and chunk.page_content.strip()
    ]

    print(f"Chunks created: {len(chunks)}")

    return chunks


def prepare_chunks(chunks):
    ids = []

    for i, chunk in enumerate(chunks):
        source_path = chunk.metadata.get("source", "unknown_file")
        source_file = Path(source_path).name
        page_number = chunk.metadata.get("page", 0) + 1

        chunk.metadata["source_file"] = source_file
        chunk.metadata["page_number"] = page_number

        ids.append(f"{source_file}_page_{page_number}_chunk_{i}")

    return chunks, ids


def create_vector_store():
    documents = load_pdf_documents()
    chunks = split_documents(documents)
    chunks, ids = prepare_chunks(chunks)

    embeddings = get_embedding_function()

    test_embedding = embeddings.embed_query("test")
    print(f"Embedding test vector size: {len(test_embedding)}")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_LOCATION),
        embedding_function=embeddings
    )

    vector_store.add_documents(
        documents=chunks,
        ids=ids
    )

    print(f"Successfully added {len(chunks)} chunks to Chroma DB.")


def get_retriever():
    embeddings = get_embedding_function()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_LOCATION),
        embedding_function=embeddings
    )

    return vector_store.as_retriever(
        search_kwargs={"k": 5}
    )