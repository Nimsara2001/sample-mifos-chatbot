# streamlit_app.py
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import streamlit as st
import time
import logging
import os
import threading
import concurrent.futures

from execute_agent import execute_all_agents

# Ensure RAGController can be imported
try:
    from rag_controller import RAGController
except ImportError:
    st.error("Failed to import RAGController. Make sure rag_controller.py is in the same directory.")
    st.stop()
except FileNotFoundError as e:
    st.error(f"Database Error: {e}. Please run the data processing step first (e.g., python main.py --reprocess).")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during RAG Controller initialization: {e}")
    logging.error("Error initializing RAGController", exc_info=True)
    st.stop()

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Mifos RAG Chat", layout="wide")
st.image("static/logo.png", width=100)
st.title("Mifos Intelligent Documentation Assistant")
st.caption("Ask questions about Mifos based on Confluence and GitHub documentation.")


# def run_async_in_thread(async_func):
#     result = None
#     exception = None
#
#     def run_in_thread():
#         nonlocal result, exception
#         try:
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
#             result = loop.run_until_complete(async_func())
#             loop.close()
#         except Exception as e:
#             exception = e
#
#     thread = threading.Thread(target=run_in_thread)
#     thread.start()
#     thread.join()
#
#     if exception:
#         raise exception
#     return result


# --- Initialize RAG Controller ---
# Cache the controller to avoid re-initializing on every interaction
@st.cache_resource
def get_rag_controller():
    try:
        controller = RAGController()
        return controller
    except FileNotFoundError as e:
        st.error(
            f"Database Error: {e}. Please run the data processing step first (e.g., `python main.py --reprocess`).")
        logging.error(f"Database Error: {e}")
        return None  # Return None to indicate failure
    except Exception as e:
        st.error(f"Failed to initialize RAG Controller: {e}")
        logging.error("Error getting RAG controller", exc_info=True)
        return None  # Return None to indicate failure


rag_controller = get_rag_controller()

# --- Session State Management ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial assistant message if controller is ready
    if rag_controller:
        st.session_state.messages.append(
            {"role": "assistant", "content": "Hello! How can I help you with Mifos documentation today?"}
        )
    else:
        st.session_state.messages.append(
            {"role": "assistant", "content": "System initialization failed. Please check logs and prerequisites."}
        )

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = [{"role": "assistant", "content": "Chat history cleared. Ask me a new question!"}]
        if rag_controller:
            rag_controller.clear_history()
        st.rerun()

    # Add Update Knowledge Base button
    if st.button("Update Knowledge Base"):
        with st.spinner("Updating knowledge base... This may take a few minutes."):
            try:
                # Run the async function
                print("Updating knowledge base.......................")
                asyncio.run(execute_all_agents())
                print("Knowledge base updated successfully...........")
                st.success("Knowledge base updated successfully!")
                # Reinitialize the RAG controller to use new data
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Failed to update knowledge base: {str(e)}")
                logging.error("Error updating knowledge base", exc_info=True)

    st.divider()
    st.subheader("Info")
    st.markdown(
        """
        This chatbot uses Model Context Protocol (MCP) Agents 
        to retrieve and answer questions using:
        - Confluence Agent: Fetches Mifos documentation
        - GitHub Agent: Retrieves repository content

        Retrieved content is processed through RAG and stored in ChromaDB.
        OpenAI (`gpt-4-mini`) is used for response generation.
        """
    )

# --- Main Chat Interface ---

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources if available (and if it was an assistant message with sources)
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("Sources Used", expanded=False):
                # Display unique sources
                unique_sources = sorted(list(set(message["sources"])))
                for source in unique_sources:
                    st.caption(f"- {source}")

# Accept user input
if prompt := st.chat_input("Ask your question about Mifos..."):
    if not rag_controller:
        st.error("RAG system is not available. Cannot process query.")
    else:
        # Add user message to chat history and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display thinking indicator and process query
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            start_time = time.time()

            try:
                # Get response from RAG controller
                response = rag_controller.ask(prompt)
                answer = response.get("answer", "Sorry, I couldn't get an answer.")
                source_docs = response.get("source_documents", [])
                end_time = time.time()

                # Extract filenames from source documents metadata
                sources = []
                if source_docs:
                    for doc in source_docs:
                        filename = doc.metadata.get('filename', doc.metadata.get('source', None))
                        if filename:
                            sources.append(os.path.basename(filename))  # Get only filename

                # Display the actual response
                message_placeholder.markdown(answer)
                logging.info(f"Response generated in {end_time - start_time:.2f} seconds.")

                # Store assistant response and sources in history
                assistant_message = {"role": "assistant", "content": answer, "sources": sources}
                st.session_state.messages.append(assistant_message)

                # Optionally display sources again if needed (already done above the input box)
                # if sources:
                #     with st.expander("Sources Used", expanded=False):
                #         unique_sources = sorted(list(set(sources)))
                #         for source in unique_sources:
                #             st.caption(f"- {source}")

            except Exception as e:
                logging.error(f"Error processing user query: {e}", exc_info=True)
                error_message = f"Sorry, an error occurred while processing your request: {e}"
                message_placeholder.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
