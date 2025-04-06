# data_processor.py
import os
import re
import shutil
from typing import List, Tuple
import logging

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import chromadb # Import chromadb client library

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
CONFLUENCE_DIR = "mifos_confluence_mds"
GITHUB_DIR = "mifos_github_mds"
CHROMA_DB_PATH = "./chroma_db"
CONFLUENCE_COLLECTION_NAME = "mifos_confluence"
GITHUB_COLLECTION_NAME = "mifos_github"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Good default, runs locally

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Helper Functions ---

def clean_document_content(documents: List[Document]) -> List[Document]:
    """Basic cleaning: replace multiple newlines/spaces."""
    logging.info(f"Cleaning {len(documents)} documents.")
    cleaned_docs = []
    for doc in documents:
        # Replace multiple newlines with one
        content = re.sub(r'\n\s*\n', '\n', doc.page_content)
        # Replace multiple spaces with one (but preserve line breaks)
        content = re.sub(r' {2,}', ' ', content)
        # Remove leading/trailing whitespace from each line
        content = "\n".join([line.strip() for line in content.split('\n')])
        # Remove excessive blank lines at start/end of doc
        content = content.strip()

        if content: # Only add if content remains after cleaning
           cleaned_doc = Document(page_content=content, metadata=doc.metadata)
           cleaned_docs.append(cleaned_doc)
        else:
            logging.warning(f"Document became empty after cleaning: {doc.metadata.get('source', 'N/A')}")

    logging.info(f"Finished cleaning. {len(cleaned_docs)} documents remain.")
    return cleaned_docs

def load_markdown_files(directory: str) -> List[Document]:
    """Loads markdown files from a directory."""
    if not os.path.isdir(directory):
        logging.error(f"Directory not found: {directory}")
        return []
    logging.info(f"Loading markdown files from: {directory}")
    # Use UnstructuredMarkdownLoader for better parsing of complex markdown
    loader = DirectoryLoader(
        directory,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
        use_multithreading=True, # Speed up loading
        silent_errors=True # Log errors but continue
    )
    try:
        documents = loader.load()
        # Ensure 'source' metadata is properly set (sometimes loader adds full path)
        for doc in documents:
            if 'source' in doc.metadata:
                doc.metadata['filename'] = os.path.basename(doc.metadata['source'])
        logging.info(f"Loaded {len(documents)} documents from {directory}.")
        return documents
    except Exception as e:
        logging.error(f"Error loading documents from {directory}: {e}")
        return []

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Splits documents into manageable chunks."""
    logging.info(f"Chunking {len(documents)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True, # Helpful for potential future reference
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split into {len(chunks)} chunks.")
    return chunks

def create_and_store_embeddings(
    documents: List[Document],
    collection_name: str,
    embedding_model: HuggingFaceEmbeddings,
    db_path: str
):
    """Creates embeddings and stores them in ChromaDB."""
    if not documents:
        logging.warning(f"No documents provided for collection '{collection_name}', skipping embedding.")
        return

    logging.info(f"Creating and storing embeddings for collection: '{collection_name}'")

    # Use Langchain's Chroma wrapper for easier integration
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=db_path,
        # Optional: Add specific client settings if needed
        # client_settings=chromadb.config.Settings(...)
    )
    # Persisting is handled automatically by from_documents with persist_directory
    # vectorstore.persist() # Not strictly needed here but good practice if adding incrementally later
    logging.info(f"Successfully stored embeddings for '{collection_name}' in {db_path}")


def process_data(force_reprocess: bool = False):
    """Main function to orchestrate data loading, processing, and storage."""
    logging.info("Starting data processing...")

    # Check if ChromaDB exists and reprocessing is not forced
    if os.path.exists(CHROMA_DB_PATH) and not force_reprocess:
        # Quick check if collections seem to exist
        try:
            client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            client.get_collection(CONFLUENCE_COLLECTION_NAME)
            client.get_collection(GITHUB_COLLECTION_NAME)
            logging.info("ChromaDB already exists and collections found. Skipping processing.")
            logging.info("Use the --reprocess flag in main.py to force reprocessing.")
            return # Skip processing
        except Exception as e: # Catch potential exceptions if collections don't exist
            logging.warning(f"ChromaDB directory exists, but collections might be missing or invalid ({e}). Reprocessing...")
            if os.path.exists(CHROMA_DB_PATH):
                 logging.info(f"Removing existing ChromaDB directory: {CHROMA_DB_PATH}")
                 shutil.rmtree(CHROMA_DB_PATH) # Remove dir to start fresh

    elif os.path.exists(CHROMA_DB_PATH) and force_reprocess:
        logging.info(f"Force reprocessing requested. Removing existing ChromaDB directory: {CHROMA_DB_PATH}")
        shutil.rmtree(CHROMA_DB_PATH)

    # Ensure ChromaDB directory exists for the new instance
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    # --- Initialize Embedding Model ---
    logging.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    # Use device='cuda' if GPU is available and configured, else 'cpu'
    # Consider adding error handling for model loading
    try:
         embedding_model = HuggingFaceEmbeddings(
             model_name=EMBEDDING_MODEL_NAME,
             model_kwargs={'device': 'cpu'} # Or 'cuda' if available
         )
    except Exception as e:
        logging.error(f"Failed to load embedding model {EMBEDDING_MODEL_NAME}: {e}")
        logging.error("Please ensure sentence-transformers and necessary model dependencies (like PyTorch or TensorFlow) are installed.")
        return # Stop processing if embedding model fails

    # --- Process Confluence Data ---
    confluence_docs = load_markdown_files(CONFLUENCE_DIR)
    if confluence_docs:
        confluence_docs_cleaned = clean_document_content(confluence_docs)
        confluence_chunks = chunk_documents(confluence_docs_cleaned)
        create_and_store_embeddings(
            confluence_chunks,
            CONFLUENCE_COLLECTION_NAME,
            embedding_model,
            CHROMA_DB_PATH
        )
    else:
        logging.warning(f"No documents found or loaded from {CONFLUENCE_DIR}. Skipping Confluence processing.")

    # --- Process GitHub Data ---
    github_docs = load_markdown_files(GITHUB_DIR)
    if github_docs:
        github_docs_cleaned = clean_document_content(github_docs)
        github_chunks = chunk_documents(github_docs_cleaned)
        create_and_store_embeddings(
            github_chunks,
            GITHUB_COLLECTION_NAME,
            embedding_model,
            CHROMA_DB_PATH
        )
    else:
        logging.warning(f"No documents found or loaded from {GITHUB_DIR}. Skipping GitHub processing.")

    logging.info("Data processing finished.")

# Allow running this script directly for reprocessing if needed
if __name__ == "__main__":
    print("Running data processor directly...")
    process_data(force_reprocess=True) # Example: Force reprocess when run directly
    print("Data processing complete.")