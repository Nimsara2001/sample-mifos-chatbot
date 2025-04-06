# rag_controller.py
import os
import logging
from typing import List, Dict, Any, Sequence, Tuple

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# Import constants from data_processor
from data_processor import (
    CHROMA_DB_PATH,
    CONFLUENCE_COLLECTION_NAME,
    GITHUB_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Custom Retriever for Multiple Collections ---
class MultipleCollectionRetriever(BaseRetriever):
    """
    A custom retriever that fetches documents from multiple ChromaDB collections.
    """
    vectorstores: List[Chroma]
    search_type: str = "similarity"
    search_kwargs: dict = {"k": 5}  # Number of docs to fetch from EACH collection

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Synchronous method to get relevant documents from all vectorstores.
        """
        all_docs = []
        for vs in self.vectorstores:
            try:
                docs = vs.similarity_search(query, **self.search_kwargs)
                logging.debug(
                    f"Retrieved {len(docs)} docs from collection '{vs._collection.name}' for query: '{query[:50]}...'")
                all_docs.extend(docs)
            except Exception as e:
                logging.error(f"Error searching collection '{vs._collection.name}': {e}")

        # Optional: Add de-duplication logic here if needed, based on content or ID
        # Simple de-duplication based on page_content:
        unique_docs = {}
        for doc in all_docs:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc

        final_docs = list(unique_docs.values())
        logging.info(f"Combined retriever fetched {len(final_docs)} unique documents.")
        return final_docs

    # Optional: Implement asynchronous version if needed for async chains
    # async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> List[Document]:
    #     # ... implementation using asyncio.gather ...
    #     pass


# --- RAG Controller Class ---
class RAGController:
    def __init__(self):
        logging.info("Initializing RAG Controller...")
        load_dotenv()  # Load OPENAI_API_KEY

        self._check_db_exists()

        self.embedding_model = self._initialize_embeddings()
        self.vectorstore_confluence, self.vectorstore_github = self._load_vectorstores()
        self.retriever = self._create_combined_retriever()
        self.llm = self._initialize_llm()
        self.memory = self._initialize_memory()
        self.chain = self._create_conversational_chain()
        logging.info("RAG Controller initialized successfully.")

    def _check_db_exists(self):
        """Check if the ChromaDB path and expected collections exist."""
        if not os.path.exists(CHROMA_DB_PATH):
            raise FileNotFoundError(
                f"ChromaDB directory not found at {CHROMA_DB_PATH}. "
                "Run the data processing step first (e.g., `python main.py --reprocess`)."
            )
        # Basic check for collection files (structure might vary slightly with Chroma versions)
        # A more robust check would involve trying to load the collections.
        if not os.path.exists(os.path.join(CHROMA_DB_PATH)):  # Chroma creates files/dirs here
            logging.warning(f"Chroma DB path {CHROMA_DB_PATH} exists, but might be empty.")

    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """Initializes the HuggingFace embedding model."""
        logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        try:
            # Ensure consistency with data processing
            return HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'}  # Or 'cuda'
            )
        except Exception as e:
            logging.error(f"Failed to load embedding model {EMBEDDING_MODEL_NAME}: {e}")
            raise RuntimeError("Could not initialize embedding model.") from e

    def _load_vectorstores(self) -> Tuple[Chroma, Chroma]:
        """Loads the Chroma vector stores from disk."""
        logging.info(f"Loading vector stores from: {CHROMA_DB_PATH}")
        try:
            vs_confluence = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=self.embedding_model,
                collection_name=CONFLUENCE_COLLECTION_NAME
            )
            vs_github = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=self.embedding_model,
                collection_name=GITHUB_COLLECTION_NAME
            )
            # Test connection by trying a dummy search (optional but good)
            _ = vs_confluence.similarity_search("test", k=1)
            _ = vs_github.similarity_search("test", k=1)
            logging.info("Vector stores loaded successfully.")
            return vs_confluence, vs_github
        except Exception as e:
            logging.error(f"Failed to load vector stores from {CHROMA_DB_PATH}: {e}")
            logging.error("Ensure the database was created correctly by the data processor.")
            raise RuntimeError("Could not load vector stores.") from e

    def _create_combined_retriever(self) -> BaseRetriever:
        """Creates a retriever that queries both collections."""
        logging.info("Creating combined retriever for Confluence and GitHub collections.")
        return MultipleCollectionRetriever(
            vectorstores=[self.vectorstore_confluence, self.vectorstore_github],
            search_kwargs={"k": 3}  # Fetch top 3 from each collection
        )

    def _initialize_llm(self) -> ChatOpenAI:
        """Initializes the ChatOpenAI LLM."""
        logging.info("Initializing OpenAI LLM...")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        try:
            # Use a cost-effective and fast model suitable for chat
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.5,  # Slightly creative but mostly factual
                openai_api_key=openai_api_key,
                max_tokens=1000
            )
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI LLM: {e}")
            raise RuntimeError("Could not initialize LLM.") from e

    def _initialize_memory(self) -> ConversationBufferMemory:
        """Initializes conversation memory."""
        logging.info("Initializing conversation memory.")
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'  # Ensure the LLM output is stored as 'answer'
        )

    def _create_conversational_chain(self) -> ConversationalRetrievalChain:
        """Creates the conversational retrieval chain."""
        logging.info("Creating ConversationalRetrievalChain...")

        # Define a custom prompt template to guide the LLM
        # This template includes placeholders for context, chat history, and the question
        _template = """Given the following conversation and relevant context from Mifos documentation (Confluence and GitHub), answer the question.
If the question cannot be answered based on the provided context, just say "I cannot answer this question based on the available Mifos documentation." Do not try to make up an answer.
Be concise and focus on the information relevant to the question. Mention the source filename(s) if relevant documents are found.

Context:
{context}

Chat History:
{chat_history}

Question: {question}
Answer:"""
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

        try:
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                return_source_documents=True,  # Return source docs used
                # combine_docs_chain_kwargs={"prompt": CONDENSE_QUESTION_PROMPT}, # Use custom prompt
                verbose=True # Set to True for debugging chain execution
            )
            logging.info("ConversationalRetrievalChain created.")
            return chain
        except Exception as e:
            logging.error(f"Failed to create ConversationalRetrievalChain: {e}")
            raise RuntimeError("Could not create RAG chain.") from e

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Queries the RAG chain with a question.

        Args:
            question: The user's question.

        Returns:
            A dictionary containing the answer and source documents.
            e.g., {'answer': 'The answer is...', 'source_documents': [doc1, doc2]}
        """
        logging.info(f"Received question: {question}")
        if not self.chain:
            logging.error("RAG chain is not initialized.")
            return {"answer": "Error: RAG system not ready.", "source_documents": []}

        try:
            # The chain handles history and context retrieval automatically
            result = self.chain.invoke({"question": question})

            # Log source documents found
            if result.get("source_documents"):
                sources = [doc.metadata.get('filename', doc.metadata.get('source', 'Unknown')) for doc in
                           result['source_documents']]
                logging.info(f"Retrieved sources: {list(set(sources))}")  # Use set for unique filenames
            else:
                logging.info("No source documents were retrieved.")

            return result  # Contains 'answer' and 'source_documents' keys
        except Exception as e:
            logging.error(f"Error during RAG chain invocation: {e}", exc_info=True)
            # Optionally reset memory on error? Depends on desired behavior.
            # self.memory.clear()
            return {"answer": f"An error occurred: {e}", "source_documents": []}

    def clear_history(self):
        """Clears the conversation memory."""
        logging.info("Clearing conversation history.")
        self.memory.clear()
