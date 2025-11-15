import streamlit as st
import os
import tempfile
import faiss  # This is the faiss-cpu library

# LlamaIndex Imports
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.prompts.prompts import SimpleInputPrompt

# --- FIXED: use Gemini (correct for LlamaIndex 0.14.8) ---
from llama_index.llms.gemini import Gemini

# Embeddings - Local CPU HuggingFace model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# --- Page Configuration ---
st.set_page_config(
    page_title="Chat with your Docs (RAG)",
    page_icon="📚",
    layout="wide"
)

# --- Sidebar for API Key ---
st.sidebar.title("🔑 Configuration")
api_key = st.sidebar.text_input("Enter your GOOGLE_API_KEY:", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    st.sidebar.success("API Key set successfully!")
else:
    st.sidebar.info("Please enter your GOOGLE_API_KEY to run.")


# --- Main Application ---
st.title("📚 Chat With Your Documents (RAG)")
st.write("Ask questions about your documents. This app uses RAG to provide factual, context-aware answers.")


# --- Document Upload ---
uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])


# --- Function to create RAG chain ---
@st.cache_resource(show_spinner="Processing document and building RAG pipeline...")
def create_chat_engine(_api_key, _uploaded_file):
    if not _api_key:
        st.error("API Key not set. Please enter it in the sidebar.")
        return None
    
    os.environ["GOOGLE_API_KEY"] = _api_key

    try:
        # 1. Set up the LLM (Gemini)
        Settings.llm = Gemini(
            model="models/gemini-2.5-flash",
            api_key=_api_key
        )

        # 2. Load local embedding model (BGE-small)
        st.info("Loading free local embedding model (BGE-small)...")
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        st.info("Embedding model loaded.")

        # 3. Load document
        data_directory = tempfile.mkdtemp()

        if _uploaded_file:
            file_path = os.path.join(data_directory, _uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(_uploaded_file.getvalue())

            st.info(f"Loading uploaded document: {_uploaded_file.name}")
        else:
            st.info("No file uploaded. Using default 'knowledge_base.txt'.")
            file_path = "knowledge_base.txt"
            if not os.path.exists(file_path):
                st.error("Default 'knowledge_base.txt' not found. Please upload a file.")
                return None
        
        loader = SimpleDirectoryReader(input_files=[file_path])
        documents = loader.load_data()

        if not documents:
            st.error("The document is empty. Please upload a readable file.")
            return None

        # 4. FAISS Vector Store
        embedding_dimension = 384  # BGE-small dimension
        faiss_index = faiss.IndexFlatL2(embedding_dimension)

        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 5. Create the vector index
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )

        # 6. Custom QA Prompt
        qa_prompt_template = (
            "You are an assistant for question-answering tasks.\n"
            "Use ONLY the context provided below. If the answer is not found, say:\n"
            "'I don't know the answer based on the provided documents.'\n\n"
            "Context:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Question: {query_str}\n"
            "Answer: "
        )

        qa_prompt = SimpleInputPrompt(qa_prompt_template)

        chat_engine = index.as_chat_engine(
            chat_mode="context",
            text_qa_template=qa_prompt
        )

        st.success("RAG pipeline built successfully!")

        # Cleanup temp folder
        if _uploaded_file:
            for f in os.listdir(data_directory):
                os.remove(os.path.join(data_directory, f))
            os.rmdir(data_directory)

        return chat_engine

    except Exception as e:
        st.error(f"Error creating RAG pipeline: {e}")
        return None


# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User Input
user_question = st.chat_input("Ask a question about your document:")

if user_question:
    if not api_key:
        st.info("Please enter your API key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.markdown(user_question)

        chat_engine = create_chat_engine(api_key, uploaded_file)

        if chat_engine:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = chat_engine.chat(user_question)
                        answer = str(response)

                        st.markdown(answer)

                        st.session_state.messages.append({"role": "assistant", "content": answer})

                        # Show retrieved context
                        with st.expander("Show Retrieved Context"):
                            st.json([node.text for node in response.source_nodes])

                    except Exception as e:
                        st.error(f"Error during chat: {e}")
