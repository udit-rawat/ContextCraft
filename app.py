import re
import streamlit as st
import os
import subprocess
from retriever import QueryEngine

# Configure environment variables for MPS and FAISS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['USE_FAISS_CPU'] = '1'


class ContentEngine:
    def __init__(self):
        self.query_engine = QueryEngine()
        try:
            self.query_engine.load_vector_store(
                'vector_store/faiss_index.idx',
                'vector_store/chunk_map.pkl',
            )
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            raise

    def check_ollama_status(self):
        """Check if Ollama is running and available"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                check=True,
                text=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def generate_response(self, query: str, context: str, chat_history: str = "") -> str:
        """Generate response using Ollama with improved error handling"""
        prompt_injection = '''You are an advanced document analysis assistant. 
        Follow these guidelines to ensure accurate and relevant responses: 
        1. Answer questions concisely based on the provided context.
        2. Do not include irrelevant details. 
        3. If insufficient context is available, indicate politely.'''

        prompt = (
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            f"Instructions:\n{prompt_injection}\n\n"
            f"Chat History:\n{chat_history}\n\n"
            "Answer:"
        )

        try:
            if not self.check_ollama_status():
                return "Error: Ollama is not running. Please start the Ollama service first."

            # Use Popen for better stream handling
            process = subprocess.Popen(
                ["ollama", "run", "llama3.2:latest"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Send prompt and get response
            stdout, stderr = process.communicate(input=prompt, timeout=60)

            # Handle the return code
            if process.returncode != 0:
                if stderr:
                    st.error(f"Ollama Error: {stderr}")
                return "An error occurred while generating the response."

            # Clean and return the response
            return self.clean_response(stdout)

        except subprocess.TimeoutExpired:
            process.kill()  # Ensure the process is terminated
            return "Error: Response generation timed out. Please try again with a simpler query."
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return "An unexpected error occurred."

    def clean_response(self, response: str) -> str:
        """Clean and format the response"""
        # Remove any ANSI escape sequences
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        cleaned_response = ansi_escape.sub('', response)

        # Remove any non-printable characters
        cleaned_response = ''.join(
            char for char in cleaned_response if char.isprintable() or char in '\n\t')

        # Remove multiple consecutive newlines
        cleaned_response = re.sub(r'\n\s*\n', '\n\n', cleaned_response)

        return cleaned_response.strip()


def main():
    st.title("WebEngine")
    st.write("Ask questions about Google, Tesla, and Uber's 10-K filings.")

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'content_engine' not in st.session_state:
        try:
            st.session_state.content_engine = ContentEngine()
        except Exception as e:
            st.error(f"Failed to initialize ContentEngine: {str(e)}")
            return

    # Check Ollama status with cleaner error message
    if not st.session_state.content_engine.check_ollama_status():
        st.error("""
        ⚠️ Ollama Service Not Available
        
        Please ensure:
        1. Ollama is installed on your system
        2. The Ollama service is running
        3. The llama2 model is downloaded (run 'ollama pull llama2')
        
        To start Ollama, open a terminal and run: ollama serve
        """)
        return

    # Chat interface
    user_input = st.text_area(
        "Your question:",
        height=100,
        placeholder="e.g., What are the main differences between Google and Tesla's revenue?"
    )

    if st.button("Ask"):
        if user_input:
            with st.spinner("Processing your question..."):
                try:
                    context = st.session_state.content_engine.query_engine.retrieve_context(
                        user_input,
                        top_k=5
                    )

                    # Generate response
                    response = st.session_state.content_engine.generate_response(
                        user_input,
                        context,
                        "\n".join(st.session_state.chat_history[-6:])
                    )

                    # Add to chat history
                    st.session_state.chat_history.extend([
                        f"User: {user_input}",
                        f"Assistant: {response}"
                    ])
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")

    # Display chat history
    if st.session_state.chat_history:
        st.write("Chat History:")
        for message in st.session_state.chat_history:
            if message.startswith("User:"):
                st.info(message)
            else:
                st.write(message)

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []


if __name__ == "__main__":
    main()
