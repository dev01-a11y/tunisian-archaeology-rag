import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
import tempfile
import os

# Set seed for consistent language detection
DetectorFactory.seed = 0

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
        color: #1f2937 !important;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        outline: none;
        color: #1f2937 !important;
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

# Language name mapping (100+ languages supported)
LANGUAGE_NAMES = {
    'en': 'English', 'fr': 'French', 'ar': 'Arabic', 'es': 'Spanish',
    'de': 'German', 'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian',
    'zh-cn': 'Chinese (Simplified)', 'zh-tw': 'Chinese (Traditional)',
    'ja': 'Japanese', 'ko': 'Korean', 'nl': 'Dutch', 'pl': 'Polish',
    'tr': 'Turkish', 'sv': 'Swedish', 'da': 'Danish', 'fi': 'Finnish',
    'no': 'Norwegian', 'cs': 'Czech', 'el': 'Greek', 'he': 'Hebrew',
    'hi': 'Hindi', 'id': 'Indonesian', 'ms': 'Malay', 'th': 'Thai',
    'vi': 'Vietnamese', 'uk': 'Ukrainian', 'ro': 'Romanian', 'hu': 'Hungarian',
    'sk': 'Slovak', 'bg': 'Bulgarian', 'hr': 'Croatian', 'sr': 'Serbian',
    'ca': 'Catalan', 'lt': 'Lithuanian', 'lv': 'Latvian', 'et': 'Estonian',
    'sl': 'Slovenian', 'af': 'Afrikaans', 'sq': 'Albanian', 'bn': 'Bengali',
    'fa': 'Persian', 'ur': 'Urdu', 'sw': 'Swahili', 'ta': 'Tamil'
}

def detect_language(text):
    """Automatically detect language from text (supports 100+ languages)"""
    try:
        detected_lang = detect(text)
        return detected_lang
    except LangDetectException:
        return 'en'
    except Exception:
        return 'en'

def translate_text(text, source_lang='auto', target_lang='en'):
    """Translate text between any languages"""
    try:
        if source_lang == target_lang or (source_lang == 'auto' and target_lang == 'en'):
            return text
        
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated = translator.translate(text)
        return translated
    except Exception as e:
        return text

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
    """Generate answer in English (will be translated later)"""
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

def rag_query(question, user_language='en'):
    """Main RAG query function with automatic multilingual support"""
    
    # Step 1: Translate question to English for database search
    question_english = question
    if user_language != 'en':
        try:
            question_english = translate_text(question, source_lang=user_language, target_lang='en')
        except:
            question_english = question
    
    # Step 2: Search database with English query
    results = retrieve_context(question_english, top_k=5)
    context, sources = format_context(results)
    
    # Step 3: Check if we have relevant sources
    if not sources:
        no_info_msg = "I don't have information about this topic. I can only answer questions about Tunisian archaeological sites like Carthage, Dougga, El Jem, Kerkouane, Sbeitla, and Bulla Regia."
        if user_language != 'en':
            no_info_msg = translate_text(no_info_msg, source_lang='en', target_lang=user_language)
        return {
            'answer': no_info_msg,
            'sources': []
        }
    
    # Step 4: Check similarity threshold
    avg_similarity = sum(s['similarity'] for s in sources) / len(sources)
    
    if avg_similarity < 0.45:
        not_found_msg = "I couldn't find relevant information about this in my database. Please ask about Tunisian archaeological sites."
        if user_language != 'en':
            not_found_msg = translate_text(not_found_msg, source_lang='en', target_lang=user_language)
        return {
            'answer': not_found_msg,
            'sources': []
        }
    
    # Step 5: Generate answer in English
    answer = generate_answer(question_english, context)
    
    # Step 6: Translate answer back to user's language
    if user_language != 'en':
        try:
            answer = translate_text(answer, source_lang='en', target_lang=user_language)
        except:
            pass
    
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
            <div style='background: #dbeafe; color: #1e40af; padding: 0.75rem; border-radius: 0.5rem; 
                        margin-top: 1rem; font-size: 0.9rem; font-weight: 600; border-left: 4px solid #3b82f6;'>
                🌍 Ask in ANY language - 100+ languages supported!
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
if 'last_audio_processed' not in st.session_state:
    st.session_state.last_audio_processed = None

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
                Explore Tunisia's archaeological wonders in any language 🌍
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Voice input section with AUTO-PROCESSING AND STATE CLEARING
    st.markdown("### 🎤 Voice Input")
    
    audio_bytes = audio_recorder(
        text="🎙️ Click to record",
        recording_color="#e74c3c",
        neutral_color="#3b82f6",
        icon_name="microphone",
        icon_size="2x",
        pause_threshold=2.0,
        key="audio_recorder"
    )
    
    # Only process if we have NEW audio (not already processed)
    if audio_bytes and audio_bytes != st.session_state.last_audio_processed:
        st.audio(audio_bytes, format="audio/wav")
        
        # Mark this audio as processed
        st.session_state.last_audio_processed = audio_bytes
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        # Transcribe
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(tmp_file_path) as source:
                audio_data = recognizer.record(source)
            
            text = recognizer.recognize_google(audio_data, language='en-US')
            st.success(f"✅ Transcribed: **{text}**")
            
            # AUTO-PROCESS THE QUESTION
            detected_lang = detect_language(text)
            lang_name = LANGUAGE_NAMES.get(detected_lang, detected_lang.upper())
            
            lang_flags = {
                'en': '🇬🇧', 'fr': '🇫🇷', 'ar': '🇸🇦', 'es': '🇪🇸',
                'de': '🇩🇪', 'it': '🇮🇹', 'pt': '🇵🇹', 'ru': '🇷🇺',
                'zh-cn': '🇨🇳', 'ja': '🇯🇵', 'ko': '🇰🇷', 'nl': '🇳🇱',
                'tr': '🇹🇷', 'pl': '🇵🇱', 'sv': '🇸🇪', 'hi': '🇮🇳'
            }
            flag = lang_flags.get(detected_lang, '🌍')
            
            st.info(f"{flag} Detected language: **{lang_name}**")
            
            with st.spinner("🤔 Searching through ancient texts..."):
                result = rag_query(text, detected_lang)
                
                st.markdown("### 📝 Answer")
                
                cant_answer_keywords = [
                    "can only answer", "couldn't find", "ne peux", "n'ai pas",
                    "فقط", "لا أملك", "solo puedo", "nur", "alleen", "только"
                ]
                
                is_rejection = any(keyword in result['answer'].lower() for keyword in cant_answer_keywords)
                
                if is_rejection:
                    st.warning(result['answer'])
                else:
                    st.success(result['answer'])
                
                if result['sources']:
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
                
                # Add to history
                st.session_state.history.append({
                    'question': text,
                    'answer': result['answer'],
                    'sources': result['sources'],
                    'language': lang_name
                })
            
        except sr.UnknownValueError:
            st.error("❌ Could not understand audio")
        except Exception as e:
            st.error(f"❌ Error: {e}")
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    elif audio_bytes:
        # Audio already processed, just show it
        st.audio(audio_bytes, format="audio/wav")
        st.info("💡 Audio already processed. Record a new question or type below.")
    
    st.markdown("### 💬 Or Type Your Question")
    
    question = st.text_input(
        "",
        value=st.session_state.question,
        placeholder="e.g., What is Carthage? | Qu'est-ce que Carthage? | ما هي قرطاج؟ | ¿Qué es Cartago? | Was ist Karthago?",
        label_visibility="collapsed"
    )
    
    ask_button = st.button("🚀 Ask Question", type="primary", use_container_width=True)
    
    if ask_button and question:
        # Detect language automatically
        detected_lang = detect_language(question)
        
        # Get language name
        lang_name = LANGUAGE_NAMES.get(detected_lang, detected_lang.upper())
        
        # Show detected language with flag
        lang_flags = {
            'en': '🇬🇧', 'fr': '🇫🇷', 'ar': '🇸🇦', 'es': '🇪🇸',
            'de': '🇩🇪', 'it': '🇮🇹', 'pt': '🇵🇹', 'ru': '🇷🇺',
            'zh-cn': '🇨🇳', 'ja': '🇯🇵', 'ko': '🇰🇷', 'nl': '🇳🇱',
            'tr': '🇹🇷', 'pl': '🇵🇱', 'sv': '🇸🇪', 'hi': '🇮🇳'
        }
        flag = lang_flags.get(detected_lang, '🌍')
        
        st.info(f"{flag} Detected language: **{lang_name}**")
        
        with st.spinner("🤔 Searching through ancient texts..."):
            result = rag_query(question, detected_lang)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📝 Answer")
            
            # Check for various "can't answer" messages in different languages
            cant_answer_keywords = [
                "can only answer", "couldn't find", "ne peux", "n'ai pas",
                "فقط", "لا أملك", "solo puedo", "nur", "alleen", "только"
            ]
            
            is_rejection = any(keyword in result['answer'].lower() for keyword in cant_answer_keywords)
            
            if is_rejection:
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
                'sources': result['sources'],
                'language': lang_name
            })
            
            st.session_state.question = ""

# History
if st.session_state.history:
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### 📜 Recent Conversations")
        
        for idx, item in enumerate(reversed(st.session_state.history[-3:])):
            lang_display = f" ({item.get('language', 'Unknown')})" if 'language' in item else ""
            with st.expander(f"💭 {item['question']}{lang_display}", expanded=(idx==0)):
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
            🇹🇳 Preserving Tunisia's Heritage Through Technology | 🌍 100+ Languages Supported | 🎤 Voice Input Enabled
        </p>
    </div>
""", unsafe_allow_html=True)
