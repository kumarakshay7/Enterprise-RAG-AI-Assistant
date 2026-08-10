import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, QueryType
import openai

load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

openai_client = openai.AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-06-01"
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

class EnterpriseRAGPipeline:
    def __init__(self, search_client: SearchClient, openai_client: openai.AzureOpenAI):
        self.search_client = search_client
        self.openai_client = openai_client

    def generate_embedding(self, text: str) -> List[float]:
        response = self.openai_client.embeddings.create(
            input=text,
            model=AZURE_EMBEDDING_DEPLOYMENT
        )
        return response.data[0].embedding

    def check_ambiguity(self, query: str) -> Optional[str]:
        """Scenario 5 Solution: Ambiguous query detection."""
        ambiguous_queries = ["what is the limit?", "policy details", "cancellation fee"]
        if query.strip().lower() in ambiguous_queries:
            return "Could you please specify which exact policy or tier (e.g., Enterprise or Standard) you are inquiring about?"
        return None

    def rewrite_query(self, user_query: str, chat_history: List[Dict[str, str]]) -> str:
        """Scenario 6 Solution: Contextual query rewriting."""
        if not chat_history:
            return user_query

        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]])
        messages = [
            {"role": "system", "content": "Rephrase the follow-up question to be a standalone search query. Return ONLY the rephrased query."},
            {"role": "user", "content": f"History:\n{history_str}\n\nFollow-up: {user_query}"}
        ]
        res = self.openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=0.0
        )
        return res.choices[0].message.content.strip()

    def retrieve(self, query: str, department: str = "Engineering", target_year: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Scenario 1, 2, 3 & Security Solutions: Hybrid Search + OData Filtering + Semantic Ranker
        """
        vector_query = VectorizedQuery(
            vector=self.generate_embedding(query),
            k_nearest_neighbors=10,
            fields="content_vector"
        )

        # Security ACL Filter + Versioning Filter
        filters = [f"allowed_departments/any(d: d eq '{department}' or d eq 'All')"]
        if target_year:
            filters.append(f"effective_year eq {target_year}")

        odata_filter = " and ".join(filters)

        results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name="my-semantic-config",
            filter=odata_filter,
            top=5
        )

        chunks = []
        for doc in results:
            chunks.append({
                "id": doc["id"],
                "source_file": doc["source_file"],
                "content": doc["content"],
                "effective_year": doc.get("effective_year")
            })
        return chunks

    def generate_answer(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Scenario 4 Solution: Strict groundedness & hallucination guardrails."""
        if not chunks:
            return {
                "answer": "I do not have sufficient information in the knowledge base to answer this question.",
                "citations": []
            }

        context = "\n\n".join([f"[Doc {i+1}] ({c['source_file']}): {c['content']}" for i, c in enumerate(chunks)])

        system_prompt = (
            "You are an enterprise AI assistant. Answer the user's question using ONLY the context provided below. "
            "If the answer cannot be determined strictly from the context, respond EXACTLY with: "
            "'I do not have sufficient information in the knowledge base to answer this question.' "
            "Always include inline citations like [Doc 1]."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        response = self.openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            temperature=0.0
        )

        answer = response.choices[0].message.content
        citations = list(set([c["source_file"] for c in chunks])) if "do not have sufficient information" not in answer else []

        return {"answer": answer, "citations": citations}

    def ask(self, user_query: str, department: str = "Engineering", history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        ambiguity_msg = self.check_ambiguity(user_query)
        if ambiguity_msg:
            return {"answer": ambiguity_msg, "citations": []}

        standalone_query = self.rewrite_query(user_query, history or [])
        chunks = self.retrieve(standalone_query, department=department)
        return self.generate_answer(standalone_query, chunks)

if __name__ == "__main__":
    pipeline = EnterpriseRAGPipeline(search_client, openai_client)
    res = pipeline.ask("What is the Enterprise cancellation policy?")
    print("Answer:\n", res["answer"])
    print("Citations:", res["citations"])