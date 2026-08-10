import os
import pypdf
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch
)
from azure.search.documents import SearchClient
import openai

load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_EMBEDDING_DEPLOYMENT]):
    missing = [
        name for name, value in [
            ("AZURE_SEARCH_ENDPOINT", AZURE_SEARCH_ENDPOINT),
            ("AZURE_SEARCH_KEY", AZURE_SEARCH_KEY),
            ("AZURE_SEARCH_INDEX", AZURE_SEARCH_INDEX),
            ("AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT),
            ("AZURE_OPENAI_KEY", AZURE_OPENAI_KEY),
            ("AZURE_EMBEDDING_DEPLOYMENT", AZURE_EMBEDDING_DEPLOYMENT),
        ] if not value
    ]
    raise EnvironmentError(f"Missing required environment variable(s): {', '.join(missing)}")

openai_client = openai.AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-06-01"
)

def create_azure_index():
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="source_file", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="effective_year", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SearchField(name="allowed_departments", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=3072,
            vector_search_profile_name="myHnswProfile"
        )
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
        profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")]
    )

    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")]
        )
    )

    semantic_search = SemanticSearch(configurations=[semantic_config])

    index = SearchIndex(
        name=AZURE_SEARCH_INDEX,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search
    )

    index_client.create_or_update_index(index)
    print(f"Index '{AZURE_SEARCH_INDEX}' created/updated successfully.")

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Handles Scenario 1: Optimized chunking with overlap."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += (chunk_size - overlap)
    return chunks

def extract_pdf_text(filepath: str) -> str:
    reader = pypdf.PdfReader(filepath)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def index_documents_from_folder(folder_path: str = "./documents"):
    create_azure_index()
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )

    documents_to_upload = []
    doc_counter = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            text = extract_pdf_text(filepath)
            chunks = chunk_text(text)

            # Metadata extraction logic for Scenario 3
            year = 2026 if "2026" in filename else 2024

            for idx, chunk in enumerate(chunks):
                doc_counter += 1
                # Generate embedding
                response = openai_client.embeddings.create(
                    input=chunk,
                    model=AZURE_EMBEDDING_DEPLOYMENT
                )
                embedding = response.data[0].embedding

                doc_id = f"doc_{doc_counter}"
                documents_to_upload.append({
                    "id": doc_id,
                    "doc_id": filename,
                    "source_file": filename,
                    "content": chunk,
                    "effective_year": year,
                    "allowed_departments": ["Engineering", "HR", "Finance", "All"],
                    "content_vector": embedding
                })

    if documents_to_upload:
        search_client.upload_documents(documents_to_upload)
        print(f"Uploaded {len(documents_to_upload)} chunks to Azure AI Search.")

if __name__ == "__main__":
    index_documents_from_folder()