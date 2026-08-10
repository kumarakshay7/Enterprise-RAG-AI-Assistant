import time
import json
from typing import Dict, Any
from rag_app import EnterpriseRAGPipeline, search_client, openai_client

# Updated Benchmark Dataset using the actual uploaded assignment PDF
EVAL_DATASET = [
    {
        "id": "Q1",
        "category": "Straightforward",
        "question": "What is the recommended Azure Stack for building the RAG pipeline?",
        "expected_doc": "Senior Al Engineer - Technical Assignment Task - Google Docs.pdf",
        "expected_keywords": ["Azure OpenAI", "Azure AI Search", "Foundry"]
    },
    {
        "id": "Q2",
        "category": "Multi-Section / Deliverables",
        "question": "What deliverables are required for submission?",
        "expected_doc": "Senior Al Engineer - Technical Assignment Task - Google Docs.pdf",
        "expected_keywords": ["GitHub", "Video", "Architecture"]
    },
    {
        "id": "Q3",
        "category": "No Answer (Hallucination Guardrail)",
        "question": "What is the company policy on orbital space tourism allowance?",
        "expected_doc": None,
        "expected_keywords": ["sufficient information"]
    },
    {
        "id": "Q4",
        "category": "Ambiguous Query",
        "question": "What is the limit?",
        "expected_doc": None,
        "expected_keywords": ["specify", "Enterprise or Standard"]
    }
]

def evaluate_pipeline() -> Dict[str, Any]:
    pipeline = EnterpriseRAGPipeline(search_client, openai_client)
    total_queries = len(EVAL_DATASET)
    hit_count = 0
    grounded_count = 0
    total_latency = 0.0

    print("\n==================================================")
    print("      RUNNING AZURE RAG EVALUATION BENCHMARK      ")
    print("==================================================\n")

    for idx, test_case in enumerate(EVAL_DATASET, 1):
        start_time = time.time()
        
        res = pipeline.ask(test_case["question"], department="Engineering")
        
        latency = time.time() - start_time
        total_latency += latency
        
        answer = res.get("answer", "")
        citations = res.get("citations", [])

        # 1. Retrieval Hit Rate Evaluation
        if test_case["expected_doc"]:
            hit = any(test_case["expected_doc"].lower() in c.lower() for c in citations)
        else:
            hit = (len(citations) == 0)
        
        if hit:
            hit_count += 1

        # 2. Answer Groundedness / Correctness
        grounded = any(kw.lower() in answer.lower() for kw in test_case["expected_keywords"])
        if grounded:
            grounded_count += 1

        print(f"[{idx}/{total_queries}] Question: {test_case['question']}")
        print(f"  ├─ Latency: {latency:.2f}s")
        print(f"  ├─ Retrieval Hit: {'✅' if hit else '❌'}")
        print(f"  └─ Groundedness: {'✅' if grounded else '❌'}\n")

    results = {
        "Total Test Cases": total_queries,
        "Retrieval Hit Rate": f"{(hit_count / total_queries) * 100:.2f}%",
        "Answer Groundedness": f"{(grounded_count / total_queries) * 100:.2f}%",
        "Average System Latency": f"{total_latency / total_queries:.2f}s"
    }

    print("==================================================")
    print("               EVALUATION SUMMARY                 ")
    print("==================================================")
    print(json.dumps(results, indent=4))
    return results

if __name__ == "__main__":
    evaluate_pipeline()