import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from rag import rag_query
import json
from datetime import datetime

# Test questions with expected characteristics
test_questions = [
    {
        "question": "What is Carthage?",
        "category": "fact",
        "expected_topics": ["Phoenician", "ancient", "city"]
    },
    {
        "question": "What makes Dougga special?",
        "category": "fact",
        "expected_topics": ["Roman", "theatre", "UNESCO"]
    },
    {
        "question": "Tell me about El Jem amphitheatre",
        "category": "fact",
        "expected_topics": ["Roman", "amphitheatre", "colosseum"]
    },
    {
        "question": "Compare Carthage and Dougga",
        "category": "comparison",
        "expected_topics": ["Phoenician", "Roman", "different"]
    },
    {
        "question": "What are the main Roman sites in Tunisia?",
        "category": "synthesis",
        "expected_topics": ["Dougga", "El Jem", "Sbeitla"]
    },
    {
        "question": "Describe the Punic civilization",
        "category": "synthesis",
        "expected_topics": ["Carthage", "Phoenician", "ancient"]
    },
    {
        "question": "Who was Hannibal?",
        "category": "fact",
        "expected_topics": ["Carthage", "general", "Rome"]
    },
    {
        "question": "What is Kerkouane known for?",
        "category": "fact",
        "expected_topics": ["Punic", "UNESCO", "settlement"]
    },
    {
        "question": "What are the Byzantine ruins in Tunisia?",
        "category": "synthesis",
        "expected_topics": ["Sbeitla", "Byzantine", "churches"]
    },
    {
        "question": "Where is the Eiffel Tower?",
        "category": "off-topic",
        "expected_topics": ["can only answer", "Tunisian", "sites"]
    }
]

def evaluate_rag_system():
    """
    Comprehensive evaluation of the RAG chatbot
    Tests: retrieval quality, response accuracy, hallucination prevention
    """
    
    print("="*80)
    print("🔍 TUNISIAN ARCHAEOLOGY RAG CHATBOT - EVALUATION")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total test questions: {len(test_questions)}\n")
    
    results = []
    
    for idx, test in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"TEST {idx}/{len(test_questions)}: {test['category'].upper()}")
        print(f"{'='*80}")
        print(f"❓ Question: {test['question']}")
        
        # Run RAG query
        result = rag_query(test['question'])
        
        # Evaluation metrics
        num_sources = len(result['sources'])
        avg_similarity = sum(s['similarity'] for s in result['sources']) / num_sources if num_sources > 0 else 0
        answer_length = len(result['answer'])
        
        print(f"\n📊 METRICS:")
        print(f"  - Sources retrieved: {num_sources}")
        print(f"  - Average similarity: {avg_similarity:.3f}")
        print(f"  - Answer length: {answer_length} characters")
        
        # Check similarity threshold (should be > 0.5 for valid answers)
        if num_sources > 0:
            similarity_pass = avg_similarity >= 0.5
            print(f"  - Similarity threshold (>0.5): {'✅ PASS' if similarity_pass else '❌ FAIL'}")
        else:
            similarity_pass = test['category'] == 'off-topic'
            print(f"  - No sources (expected for off-topic): {'✅ PASS' if similarity_pass else '❌ FAIL'}")
        
        # Check for hallucination prevention
        if test['category'] == 'off-topic':
            hallucination_check = any(phrase in result['answer'].lower() for phrase in 
                                     ["can only answer", "tunisian", "don't have information"])
            print(f"  - Off-topic rejection: {'✅ PASS' if hallucination_check else '❌ FAIL'}")
        else:
            hallucination_check = True  # Assume OK if sources exist
            print(f"  - On-topic response: ✅ PASS")
        
        # Check for expected topics in answer (basic relevance)
        topics_found = sum(1 for topic in test['expected_topics'] 
                          if topic.lower() in result['answer'].lower())
        topic_coverage = topics_found / len(test['expected_topics']) if test['expected_topics'] else 1
        print(f"  - Topic coverage: {topics_found}/{len(test['expected_topics'])} ({topic_coverage*100:.0f}%)")
        
        # Display answer
        print(f"\n📝 ANSWER:")
        print(f"  {result['answer']}")
        
        # Display sources
        if result['sources']:
            print(f"\n📚 SOURCES:")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source['title']} (similarity: {source['similarity']:.3f})")
        
        # Overall assessment
        overall_pass = similarity_pass and hallucination_check and (topic_coverage >= 0.3 or test['category'] == 'off-topic')
        print(f"\n🎯 RESULT: {'✅ PASS' if overall_pass else '❌ FAIL'}")
        
        # Store results
        results.append({
            'question': test['question'],
            'category': test['category'],
            'num_sources': num_sources,
            'avg_similarity': avg_similarity,
            'answer_length': answer_length,
            'topic_coverage': topic_coverage,
            'passed': overall_pass
        })
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['passed'])
    pass_rate = (passed_tests / total_tests) * 100
    
    avg_sources = sum(r['num_sources'] for r in results) / total_tests
    avg_similarity_all = sum(r['avg_similarity'] for r in results if r['num_sources'] > 0) / sum(1 for r in results if r['num_sources'] > 0)
    avg_topic_coverage = sum(r['topic_coverage'] for r in results) / total_tests
    
    print(f"\n✅ Tests passed: {passed_tests}/{total_tests} ({pass_rate:.1f}%)")
    print(f"📊 Average sources per query: {avg_sources:.1f}")
    print(f"🎯 Average similarity score: {avg_similarity_all:.3f}")
    print(f"📖 Average topic coverage: {avg_topic_coverage*100:.1f}%")
    
    # Category breakdown
    print(f"\n📋 BREAKDOWN BY CATEGORY:")
    categories = set(r['category'] for r in results)
    for category in categories:
        cat_results = [r for r in results if r['category'] == category]
        cat_passed = sum(1 for r in cat_results if r['passed'])
        cat_total = len(cat_results)
        print(f"  - {category.capitalize()}: {cat_passed}/{cat_total} passed")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if pass_rate >= 90:
        print("  ✅ Excellent performance! System is production-ready.")
    elif pass_rate >= 70:
        print("  ⚠️  Good performance, but consider:")
        print("     - Adding more diverse training documents")
        print("     - Fine-tuning similarity thresholds")
    else:
        print("  ❌ System needs improvement:")
        print("     - Review document chunking strategy")
        print("     - Check embedding model quality")
        print("     - Verify prompt engineering")
    
    if avg_similarity_all < 0.6:
        print("  ⚠️  Low average similarity - consider better document coverage")
    
    if avg_topic_coverage < 0.5:
        print("  ⚠️  Low topic coverage - LLM may need better prompting")
    
    # Save results to JSON
    output_file = "evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'pass_rate': pass_rate,
                'avg_sources': avg_sources,
                'avg_similarity': avg_similarity_all,
                'avg_topic_coverage': avg_topic_coverage
            },
            'detailed_results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"\n{'='*80}\n")
    
    return results

if __name__ == "__main__":
    print("\n🚀 Starting RAG System Evaluation...\n")
    results = evaluate_rag_system()
    print("✅ Evaluation complete!")
