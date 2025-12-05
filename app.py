import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# Page configuration
st.set_page_config(
    page_title="Tunisian Archaeology Chatbot",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tailwind-inspired CSS
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Tailwind bg-gray-50 equivalent */
    .main {
        background-color: #f9fafb;
    }
    
    /* Tailwind-style title */
    .main-title {
        text-align: center;
        color: #92400e;
        font-size: 3em;
        font-weight: 700;
        margin-bottom: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        letter-spacing: -0.025em;
    }
    
    /* Tailwind text-gray-600 */
    .subtitle {
        text-align: center;
        color: #4b5563;
        font-size: 1.2em;
        margin-bottom: 2em;
        font-weight: 400;
    }
    
    /* Tailwind-style buttons */
    .stButton>button {
        width: 100%;
        border-radius: 0.75rem;
        background: linear-gradient(to right, #2563eb, #1d4ed8);
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.15s ease-in-out;
    }
    
    .stButton>button:hover {
        background: linear-gradient(to right, #1d4ed8, #1e40af);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    /* Tailwind input styling - WITH BLACK TEXT */
    .stTextInput>div>div>input {
        border-radius: 0.75rem;
        border: 2px solid #e5e7eb;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.15s ease-in-out;
        background-color: white;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        color: #1f2937 !important; /* BLACK TEXT COLOR */
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        outline: none;
        color: #1f2937 !important; /* BLACK TEXT COLOR ON FOCUS */
    }
    
    /* Force all inputs to have black text */
    input {
        color: #1f2937 !important;
    }
    
    /* Tailwind sidebar */
    [data-testid="stSidebar"] {
        background-color: white;
        border-right: 1px solid #e5e7eb;
    }
    
    /* Tailwind card styling */
    .stSuccess, .stWarning, .stInfo {
        border-radius: 1rem;
        padding: 1.5rem;
        border: none;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        font-weight: 500;
    }
    
    /* Tailwind success */
    .stSuccess {
        background: linear-gradient(to bottom right, #d1fae5, #a7f3d0);
        color: #065f46;
        border-left: 4px solid #10b981;
    }
    
    /* Tailwind warning */
    .stWarning {
        background: linear-gradient(to bottom right, #fef3c7, #fde68a);
        color: #92400e;
        border-left: 4px solid #f59e0b;
    }
    
    /* Tailwind info */
    .stInfo {
        background: linear-gradient(to bottom right, #e0f2fe, #bae6fd);
        color: #075985;
        border-left: 4px solid #0ea5e9;
    }
    
    /* Tailwind expander */
    .streamlit-expanderHeader {
        background-color: #f9fafb;
        border-radius: 0.75rem;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        font-weight: 500;
        transition: all 0.15s ease-in-out;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #f3f4f6;
    }
    
    /* Tailwind progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(to right, #3b82f6, #1d4ed8);
        border-radius: 9999px;
    }
    
    /* Tailwind divider */
    hr {
        border-color: #e5e7eb;
        margin: 2rem 0;
    }
    
    /* Tailwind spinner */
    .stSpinner > div {
        border-color: #3b82f6 !important;
    }
    
    /* Placeholder text color */
    .stTextInput>div>div>input::placeholder {
        color: #9ca3af;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def load_components():
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="tunisian_archaeology")
    return embedding_model, collection

embedding_model, collection = load_components()

def retrieve_context(question, top_k=5):
    question_embedding = embedding_model.encode([question])[0]
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=top_k
    )
    return results

def format_context(results):
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    formatted_sources = []
    context_text = ""
    
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        similarity = 1 / (1 + dist)
        
        if similarity > 0.5:
            context_text += f"\n{doc}\n"
            
            source_info = {
                'number': i+1,
                'title': meta.get('title', 'Unknown'),
                'source': meta.get('source', 'Unknown'),
                'site': meta.get('site', ''),
                'similarity': similarity
            }
            formatted_sources.append(source_info)
    
    return context_text, formatted_sources

def generate_answer(question, context):
    prompt = f"""You are an expert ONLY on Tunisian archaeological sites. You can ONLY answer questions about Tunisia's ancient heritage.

Context from Tunisian archaeology database:
{context}

Question: {question}

CRITICAL INSTRUCTIONS:
- If the question is NOT about Tunisian archaeological sites, respond: "I can only answer questions about Tunisian archaeological sites."
- If the context doesn't contain relevant information, respond: "I don't have information about this in my knowledge base."
- NEVER use general world knowledge about non-Tunisian topics
- DO NOT mention source numbers
- If you can answer, write naturally in 2-4 sentences

Answer:"""

    try:
        response = ollama.generate(
            model='llama3',
            prompt=prompt,
            options={'temperature': 0.1, 'top_p': 0.9, 'num_predict': 300}
        )
        return response['response']
    except Exception as e:
        return f"Error: {str(e)}"

def rag_query(question):
    results = retrieve_context(question, top_k=5)
    context, sources = format_context(results)
    
    if not sources:
        return {
            'answer': "I don't have information about this topic. I can only answer questions about Tunisian archaeological sites like Carthage, Dougga, El Jem, Kerkouane, Sbeitla, and Bulla Regia.",
            'sources': []
        }
    
    avg_similarity = sum(s['similarity'] for s in sources) / len(sources)
    
    if avg_similarity < 0.45:
        return {
            'answer': "I couldn't find relevant information about this in my database. Please ask about Tunisian archaeological sites.",
            'sources': []
        }
    
    answer = generate_answer(question, context)
    return {'answer': answer, 'sources': sources}

# Header - Tailwind style
st.markdown("""
    <div style='background: linear-gradient(to right, #3b82f6, #8b5cf6); 
                padding: 3rem 2rem; border-radius: 1.5rem; text-align: center; 
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04); 
                margin-bottom: 2rem;'>
        <h1 style='color: white; font-size: 3em; font-weight: 700; margin: 0; 
                   text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>
            🏛️ Tunisian Archaeological Sites
        </h1>
        <p style='color: rgba(255,255,255,0.95); font-size: 1.25rem; margin: 1rem 0 0 0; font-weight: 400;'>
            Explore Tunisia's Rich Ancient Heritage with AI
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar - Tailwind cards
with st.sidebar:
    # Centered flag
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Flag_of_Tunisia.svg/320px-Flag_of_Tunisia.svg.png", width=100)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # About card - Tailwind style
    st.markdown("""
        <div style='background: white; padding: 1.5rem; border-radius: 1rem; 
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border: 1px solid #e5e7eb; margin-bottom: 1.5rem;'>
            <h3 style='color: #92400e; margin: 0 0 0.75rem 0; font-size: 1.25rem; font-weight: 600;'>🏺 About This Chatbot</h3>
            <p style='color: #4b5563; margin: 0; line-height: 1.6; font-size: 0.95rem;'>
                This AI chatbot uses <strong>RAG</strong> to provide accurate information about Tunisia's archaeological treasures ONLY.
            </p>
            <div style='background: #fef3c7; color: #92400e; padding: 0.75rem; border-radius: 0.5rem; 
                        margin-top: 1rem; font-size: 0.9rem; font-weight: 600; border-left: 4px solid #f59e0b;'>
                ⚠️ Specialized in Tunisian sites only!
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🗺️ Featured Sites")
    
    sites = {
        "🏛️": "**Carthage** - Ancient Phoenician city",
        "🎭": "**Dougga** - Best-preserved Roman town",
        "🏟️": "**El Jem** - Massive Roman amphitheatre",
        "🏺": "**Kerkouane** - Punic settlement",
        "⛪": "**Sbeitla** - Byzantine ruins",
        "🏰": "**Bulla Regia** - Underground villas"
    }
    
    for icon, desc in sites.items():
        st.markdown(f"""
            <div style='background: white; padding: 1rem; border-radius: 0.75rem; 
                        margin-bottom: 0.75rem; border: 1px solid #e5e7eb; 
                        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); transition: all 0.15s;'>
                <span style='font-size: 1.5rem; margin-right: 0.75rem;'>{icon}</span>
                <span style='color: #1f2937; font-weight: 500;'>{desc}</span>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Popular questions - Tailwind gradient
    st.markdown("""
        <div style='background: linear-gradient(to right, #3b82f6, #8b5cf6); 
                    padding: 1.25rem; border-radius: 1rem; text-align: center; 
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 1rem;'>
            <h3 style='color: white; margin: 0; font-size: 1.25rem; font-weight: 600;'>🔥 Popular Questions</h3>
            <p style='color: rgba(255,255,255,0.9); font-size: 0.875rem; margin: 0.5rem 0 0 0;'>
                Click to explore
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    example_questions = [
        "What makes Dougga special?",
        "Tell me about El Jem amphitheatre",
        "Describe the Punic civilization",
        "What is Carthage known for?",
        "Compare Carthage and Dougga"
    ]
    
    for q in example_questions:
        if st.button(f"💬 {q}", key=q, use_container_width=True):
            st.session_state.question = q
            st.rerun()

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'question' not in st.session_state:
    st.session_state.question = ""

# Main interface
col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    # Search card - Tailwind
    st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 1.5rem; 
                    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); border: 1px solid #e5e7eb; margin-bottom: 2rem;'>
            <h2 style='color: #1f2937; text-align: center; margin: 0 0 0.5rem 0; font-weight: 600;'>
                🔍 Ask Your Question
            </h2>
            <p style='text-align: center; color: #6b7280; margin: 0; font-size: 0.95rem;'>
                Explore Tunisia's archaeological wonders
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    question = st.text_input(
        "",
        value=st.session_state.question,
        placeholder="e.g., What are the main Roman monuments in Tunisia?",
        label_visibility="collapsed"
    )
    
    ask_button = st.button("🚀 Ask Question", type="primary", use_container_width=True)
    
    if ask_button and question:
        with st.spinner("🤔 Searching through ancient texts..."):
            result = rag_query(question)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📝 Answer")
            
            if "can only answer" in result['answer'].lower() or "couldn't find relevant" in result['answer'].lower():
                st.warning(result['answer'])
            else:
                st.success(result['answer'])
            
            if result['sources']:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📚 Referenced Sources")
                
                for source in result['sources']:
                    similarity_percentage = source['similarity'] * 100
                    
                    if similarity_percentage > 70:
                        color = "🟢"
                    elif similarity_percentage > 50:
                        color = "🟡"
                    else:
                        color = "🟠"
                    
                    with st.expander(f"{color} **{source['title']}** (Relevance: {similarity_percentage:.0f}%)"):
                        st.markdown(f"**📖 Source:** {source['source']}")
                        if source['site']:
                            st.markdown(f"**📍 Site:** {source['site']}")
                        st.progress(source['similarity'])
            
            st.session_state.history.append({
                'question': question,
                'answer': result['answer'],
                'sources': result['sources']
            })
            
            st.session_state.question = ""

# History
if st.session_state.history:
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### 📜 Recent Conversations")
        
        for idx, item in enumerate(reversed(st.session_state.history[-3:])):
            with st.expander(f"💭 {item['question']}", expanded=(idx==0)):
                st.markdown(f"**Question:** {item['question']}")
                st.markdown(f"**Answer:** {item['answer']}")
                st.caption(f"📚 {len(item['sources'])} sources referenced")

# Footer - Tailwind gradient
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='background: linear-gradient(to right, #3b82f6, #8b5cf6); 
                padding: 2rem; border-radius: 1.5rem; text-align: center; 
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);'>
        <p style='color: white; margin: 0; font-size: 1.125rem; font-weight: 600;'>
            Built using RAG Architecture
        </p>
        <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.95rem;'>
            🇹🇳 Preserving Tunisia's Heritage Through Technology
        </p>
    </div>
""", unsafe_allow_html=True)
