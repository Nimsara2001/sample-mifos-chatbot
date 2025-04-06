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
    
    def _score_relevance(self, doc: Document, query: str) -> float:
        """
        Score document relevance based on semantic similarity and other heuristics.
        """
        # Add basic relevance scoring - can be expanded with more sophisticated metrics
        score = 0.0
        
        # Check if query terms appear in the document
        query_terms = set(query.lower().split())
        content_terms = set(doc.page_content.lower().split())
        term_overlap = len(query_terms.intersection(content_terms)) / len(query_terms)
        score += term_overlap * 0.5  # Weight term overlap at 50%
        
        # Use metadata score if available (from ChromaDB similarity search)
        if hasattr(doc, 'metadata') and 'score' in doc.metadata:
            score += float(doc.metadata['score']) * 0.5  # Weight similarity score at 50%
            
        return score

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Get relevant documents from all vectorstores with improved relevance filtering.
        """
        all_docs = []
        for vs in self.vectorstores:
            try:
                docs = vs.similarity_search_with_score(query, **self.search_kwargs)
                for doc, score in docs:
                    doc.metadata['score'] = score  # Store similarity score
                    doc.metadata['collection'] = vs._collection.name
                    all_docs.append(doc)
            except Exception as e:
                logging.error(f"Error searching collection '{vs._collection.name}': {e}")

        # Score and rank documents
        scored_docs = [(doc, self._score_relevance(doc, query)) for doc in all_docs]
        scored_docs.sort(key=lambda x: x[1], reverse=True)  # Sort by relevance score

        # Filter out low-relevance documents
        min_score_threshold = 0.1  # Adjust this threshold as needed
        filtered_docs = [doc for doc, score in scored_docs if score > min_score_threshold]

        # De-duplicate based on content
        unique_docs = {}
        for doc in filtered_docs:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc

        final_docs = list(unique_docs.values())
        logging.info(f"Retrieved {len(final_docs)} relevant unique documents after filtering.")
        return final_docs


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
            search_kwargs={"k": 5}  # Fetch top 3 from each collection
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
                max_tokens=10000,  # Adjust based on expected answer length
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

        _template = """Given the following conversation and relevant context from Mifos documentation (Confluence and GitHub), answer the question.
If you find relevant context but are unsure about specific details, try to provide a helpful response based on the context rather than saying "I don't know".
Be specific about what information you found and what remains uncertain.
If no relevant context is found at all, then say "I cannot answer this question based on the available Mifos documentation."
Be concise and focus on the information relevant to the question. 
If relevant documents are found, cite the source filename(s) to support your answer.

Context:
{context}

Chat History:
{chat_history}

Question: {question}
Answer: """

        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

        try:
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": CONDENSE_QUESTION_PROMPT},
                verbose=True
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
