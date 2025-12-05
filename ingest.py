import os
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Initialize embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize ChromaDB
print("Initializing ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get collection
collection = client.get_or_create_collection(
    name="tunisian_archaeology",
    metadata={"description": "Tunisian archaeological sites knowledge base"}
)

def clean_text(text):
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,;:!?()\-]', '', text)
    return text.strip()

def extract_metadata(text):
    """Extract metadata from document header"""
    lines = text.split('\n')
    metadata = {
        'title': '',
        'source': '',
        'site': '',
        'topic': ''
    }
    
    for line in lines[:10]:  # Check first 10 lines
        if line.startswith('Title:'):
            metadata['title'] = line.replace('Title:', '').strip()
        elif line.startswith('Source:'):
            metadata['source'] = line.replace('Source:', '').strip()
        elif line.startswith('Site:'):
            metadata['site'] = line.replace('Site:', '').strip()
        elif line.startswith('Topic:'):
            metadata['topic'] = line.replace('Topic:', '').strip()
    
    return metadata

def chunk_text(text, chunk_size=400, overlap=50):
    """Split text into overlapping chunks by words"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.split()) > 50:  # Minimum chunk size
            chunks.append(chunk)
    
    return chunks

def process_documents():
    """Process all documents in raw_documents folder"""
    docs_folder = 'data/raw_documents'
    files = [f for f in os.listdir(docs_folder) if f.endswith('.txt')]
    
    print(f"\nProcessing {len(files)} documents...")
    
    all_chunks = []
    all_metadata = []
    all_ids = []
    
    chunk_id = 0
    
    for filename in files:
        filepath = os.path.join(docs_folder, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata
            metadata = extract_metadata(content)
            
            # Clean text (remove metadata header)
            text_lines = content.split('\n')
            main_text = '\n'.join([line for line in text_lines if not line.startswith(('Title:', 'Source:', 'Site:', 'Topic:', 'Category:'))])
            cleaned_text = clean_text(main_text)
            
            # Chunk the text
            chunks = chunk_text(cleaned_text)
            
            print(f"✓ {filename}: {len(chunks)} chunks")
            
            # Prepare data for ChromaDB
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_metadata = metadata.copy()
                chunk_metadata['filename'] = filename
                chunk_metadata['chunk_length'] = len(chunk.split())
                all_metadata.append(chunk_metadata)
                all_ids.append(f"chunk_{chunk_id}")
                chunk_id += 1
                
        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")
    
    return all_chunks, all_metadata, all_ids

def create_embeddings_and_store(chunks, metadata, ids):
    """Generate embeddings and store in ChromaDB"""
    print(f"\nGenerating embeddings for {len(chunks)} chunks...")
    
    batch_size = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_metadata = metadata[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        
        # Generate embeddings
        embeddings = embedding_model.encode(batch_chunks, show_progress_bar=False)
        
        # Store in ChromaDB
        collection.add(
            embeddings=embeddings.tolist(),
            documents=batch_chunks,
            metadatas=batch_metadata,
            ids=batch_ids
        )
        
        batch_num = (i // batch_size) + 1
        print(f"✓ Batch {batch_num}/{total_batches} stored")
    
    print("\n✅ All embeddings stored in ChromaDB!")

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("TUNISIAN ARCHAEOLOGY CHATBOT - DATA INGESTION")
    print("="*60)
    
    # Process documents
    chunks, metadata, ids = process_documents()
    
    print(f"\nTotal chunks created: {len(chunks)}")
    
    # Create embeddings and store
    create_embeddings_and_store(chunks, metadata, ids)
    
    # Verify
    count = collection.count()
    print(f"\n✅ ChromaDB collection contains {count} chunks")
    print("="*60)
