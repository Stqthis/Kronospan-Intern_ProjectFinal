"""Automated test script for JarvisMan RAG system."""
import sys
import json
from jarvisman import config as cfg
from jarvisman.llm.ollama_client import OllamaClient
from jarvisman.retrieval.rag import RAGPipeline
from jarvisman.retrieval.vector_store import VectorStore


def test_queries():
    """Run predefined queries and show results."""
    
    # Initialize components
    print("🚀 Initializing JarvisMan RAG System...\n")
    
    ollama = OllamaClient(cfg.OLLAMA_HOST)
    vector_store = VectorStore()
    rag = RAGPipeline(
        ollama=ollama,
        vector_store=vector_store,
        chat_model=cfg.DEFAULT_CHAT_MODEL,
        embed_model=cfg.DEFAULT_EMBED_MODEL,
    )
    
    # Load persisted index
    try:
        print("📚 Loading indexed data...\n")
        rag.load_persisted()
        print(f"✓ Index loaded\n")
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        print("Please build the index first using the UI.\n")
        return
    
    # Define test queries
    test_queries_list = [
        "What is the total of all type of funds of company Lignum Technologies AG in Euro equivalent as at 31.07.2023?",
        "What is the total of all type of funds in Euro (EUR) equivalent of all Croatian companies as at 31.07.2023?",
        "What is the total of all type of funds with the RBI bank in Euro (EUR) equivalent as at 31.07.2023?",
        "What is the total of all types of funds per Currency with the CaixaBank, S.A. bank as at 31.07.2023?",
        "State and/or Sort Interest rates of Funds, while also outlining the currency held.3.1 e.g. Descending sort all interest rates for all Funds in Denmark and indicate their currency, company andbank held with as at 31.07.2023",
    ]
    
    # Run queries
    print("="*70)
    print("TESTING QUERIES")
    print("="*70 + "\n")
    
    results = []
    
    for i, query in enumerate(test_queries_list, 1):
        print(f"📝 Query {i}/{len(test_queries_list)}: {query}")
        print("-" * 70)
        
        try:
            # Get answer from RAG
            answer, sources = rag.answer(query, k=5)
            
            # Store result
            result = {
                "query": query,
                "answer": answer,
                "sources": sources,
                "status": "✓ Success"
            }
            results.append(result)
            
            # Print answer
            print(f"\n💬 Answer:\n{answer}\n")
            
            # Print sources
            if sources:
                print(f"📋 Sources ({len(sources)}):")
                for src in sources:
                    print(f"  - {src['source']} {src['location']} (score: {src['score']})")
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            result = {
                "query": query,
                "answer": None,
                "sources": [],
                "status": f"❌ Error: {str(e)}"
            }
            results.append(result)
        
        print("\n" + "="*70 + "\n")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    successful = sum(1 for r in results if r["status"] == "✓ Success")
    failed = len(results) - successful
    
    print(f"\n✓ Successful: {successful}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    
    # Save results to file
    output_file = "test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved to: {output_file}\n")


if __name__ == "__main__":
    test_queries()
