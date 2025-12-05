import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# Initialize components
print("Loading components...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="tunisian_archaeology")

def retrieve_context(question, top_k=5):
    """Retrieve relevant chunks from ChromaDB"""
    question_embedding = embedding_model.encode([question])[0]
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=top_k
    )
    return results

def format_context(results):
    """Format retrieved chunks with metadata"""
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    formatted_sources = []
    context_text = ""
    
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        similarity = 1 / (1 + dist)
        print(f"  Source {i+1}: similarity={similarity:.3f}, distance={dist:.3f}")
        
        # STRICTER threshold: 0.5 instead of 0.3
        if similarity > 0.5:
            context_text += f"\n{doc}\n"
            
            source_info = {
                'number': i+1,
                'title': meta.get('title', 'Unknown'),
                'source': meta.get('source', 'Unknown'),
                'site': meta.get('site', ''),
                'filename': meta.get('filename', ''),
                'similarity': similarity
            }
            formatted_sources.append(source_info)
    
    return context_text, formatted_sources

def generate_answer(question, context):
    """Generate answer using Llama 3 via Ollama"""
    
    # IMPROVED prompt with stricter instructions
    prompt = f"""You are an expert ONLY on Tunisian archaeological sites. You can ONLY answer questions about Tunisia's ancient heritage sites like Carthage, Dougga, El Jem, Kerkouane, Sbeitla, Bulla Regia, etc.

Context from Tunisian archaeology database:
{context}

Question: {question}

CRITICAL INSTRUCTIONS:
- If the question is NOT about Tunisian archaeological sites, respond: "I can only answer questions about Tunisian archaeological sites."
- If the context doesn't contain relevant information, respond: "I don't have information about this in my knowledge base about Tunisian sites."
- NEVER use your general world knowledge about topics outside Tunisian archaeology
- DO NOT mention source numbers like [Source 1] or [Source 2]
- If you can answer, write naturally in 2-4 sentences

Answer:"""

    try:
        print("  Calling Llama 3...")
        response = ollama.generate(
            model='llama3',
            prompt=prompt,
            options={
                'temperature': 0.1,  # Lower temperature for more factual responses
                'top_p': 0.9,
                'num_predict': 300,
            }
        )
        return response['response']
    except Exception as e:
        return f"Error: {str(e)}\nMake sure Ollama is running."

def rag_query(question):
    """Complete RAG pipeline with validation"""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")
    
    # Retrieve
    print("🔍 Retrieving relevant information...")
    results = retrieve_context(question, top_k=5)
    
    # Format context
    context, sources = format_context(results)
    
    # Check if we have high-quality sources
    if not sources:
        print("⚠️  No high-quality sources found (similarity < 0.5)")
        return {
            'answer': "I don't have information about this topic in my knowledge base. I can only answer questions about Tunisian archaeological sites like Carthage, Dougga, El Jem, Kerkouane, Sbeitla, and Bulla Regia.",
            'sources': []
        }
    
    # Check average similarity
    avg_similarity = sum(s['similarity'] for s in sources) / len(sources)
    print(f"\n✓ Using {len(sources)} sources (avg similarity: {avg_similarity:.3f})\n")
    
    if avg_similarity < 0.45:
        print("⚠️  Average similarity too low - topic may be off-domain")
        return {
            'answer': "I couldn't find relevant information about this question in my database about Tunisian archaeological sites. Please ask about sites like Carthage, Dougga, El Jem, or other Tunisian heritage locations.",
            'sources': []
        }
    
    # Generate answer
    print("🤖 Generating answer with Llama 3...")
    answer = generate_answer(question, context)
    
    return {
        'answer': answer,
        'sources': sources
    }

# Test function
if __name__ == "__main__":
    # Test queries - including off-topic ones
    test_questions = [
        "What is Carthage?",
        "Tell me about El Jem amphitheatre",
        "Where is the Eiffel Tower?",  # Off-topic test
        "What are the pyramids of Egypt?",  # Off-topic test
    ]
    
    for question in test_questions:
        result = rag_query(question)
        
        print("\n📝 ANSWER:")
        print(result['answer'])
        
        if result['sources']:
            print("\n📚 SOURCES USED:")
            for source in result['sources']:
                print(f"  • {source['title']} ({source['source']}) - similarity: {source['similarity']:.3f}")
        else:
            print("\n⚠️  No sources used")
        
        print("\n" + "="*60 + "\n")
        input("Press Enter for next question...")
