import os
import streamlit as st  # type: ignore[import-not-found]
from dotenv import load_dotenv
from rag_app import EnterpriseRAGPipeline, search_client, openai_client
from ingestion import index_documents_from_folder

load_dotenv()

# Page Setup
st.set_page_config(
    page_title="Azure AI RAG Enterprise Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Azure AI RAG Enterprise Assistant")
st.caption("Production-grade Azure OpenAI + Azure AI Search Pipeline")

# Initialize Pipeline
@st.cache_resource
def get_pipeline():
    return EnterpriseRAGPipeline(search_client, openai_client)

rag_pipeline = get_pipeline()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar - Settings & Document Upload
with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload Enterprise Documents (PDF)", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if st.button("Process & Index Documents"):
        if uploaded_files:
            docs_dir = "./documents"
            os.makedirs(docs_dir, exist_ok=True)

            for uploaded_file in uploaded_files:
                file_path = os.path.join(docs_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            with st.spinner("Parsing, Chunking, Generating Embeddings & Uploading to Azure AI Search..."):
                index_documents_from_folder(docs_dir)
            st.success("Indexing Complete!")
        else:
            st.warning("Please upload at least one PDF file first.")

    st.divider()

    st.header("⚙️ Security & Context Controls")
    department = st.selectbox(
        "User Department Access",
        ["Engineering", "HR", "Finance"],
        help="Tests Scenario: Security Access-Controlled RAG"
    )

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            st.caption(f"**Citations:** {', '.join(message['citations'])}")

# User Input Field
if prompt := st.chat_input("Ask a question about your enterprise documents..."):
    # Add User Message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process Question through RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching Knowledge Base & Generating Grounded Response..."):
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            response = rag_pipeline.ask(
                user_query=prompt,
                department=department,
                history=history
            )

            answer = response["answer"]
            citations = response.get("citations", [])

            st.markdown(answer)
            if citations:
                st.caption(f"**Citations:** {', '.join(citations)}")

    # Add Assistant Message to UI State
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations
    })