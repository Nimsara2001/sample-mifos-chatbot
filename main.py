# main.py
import sys
import os
import subprocess
import argparse
import logging

# Configure logging early
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure other modules can be found if main.py is run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import data_processor
    # We don't need rag_controller here, streamlit_app imports it
except ImportError as e:
    logging.error(f"Failed to import necessary modules: {e}")
    logging.error("Please ensure data_processor.py exists and all dependencies in requirements.txt are installed.")
    sys.exit(1)
except Exception as e:
    logging.error(f"An unexpected error occurred during import: {e}")
    sys.exit(1)


def run_streamlit():
    """Runs the Streamlit application."""
    logging.info("Starting Streamlit application...")
    streamlit_app_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")

    if not os.path.exists(streamlit_app_path):
        logging.error(f"Streamlit app file not found: {streamlit_app_path}")
        sys.exit(1)

    # Use subprocess to run streamlit, ensuring it runs in the correct environment
    # This is generally more reliable than trying to run streamlit programmatically via its internal functions
    command = [sys.executable, "-m", "streamlit", "run", streamlit_app_path]

    try:
        process = subprocess.Popen(command)
        process.wait()  # Wait for streamlit process to exit
    except KeyboardInterrupt:
        logging.info("Streamlit process interrupted by user.")
        process.terminate()  # Terminate the streamlit process if Ctrl+C is pressed
    except Exception as e:
        logging.error(f"Failed to run Streamlit: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run the Mifos RAG Chat application.")
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Force reprocessing of data and recreating the vector database.",
    )
    args = parser.parse_args()

    # --- Step 1: Process Data (if needed or forced) ---
    try:
        data_processor.process_data(force_reprocess=args.reprocess)
    except Exception as e:
        logging.error(f"An error occurred during data processing: {e}", exc_info=True)
        logging.error("Data processing failed. Cannot start the application without a valid database.")
        # Optionally, you could allow starting Streamlit even if processing fails,
        # but the RAGController initialization would likely fail later.
        # For robustness, we exit here if processing fails.
        sys.exit(1)  # Exit if data processing fails critically

    # --- Step 2: Run Streamlit App ---
    run_streamlit()


if __name__ == "__main__":
    main()
