import streamlit as st
from Bio import Entrez
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai

# --- Page Config ---
st.set_page_config(page_title="PubMed AI Citation Generator", layout="centered")
st.title("🔬 PubMed AI Citation Generator")
st.markdown("Generate professional scientific citations using RAG architecture.")

# --- Sidebar for API Keys ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    email = st.text_input("PubMed Email")
    st.info("This app automatically selects the best available Gemini model for your account.")

# --- Helper: Self-Healing Model Selector ---
def get_best_model(api_key):
    """
    Bulletproof selector: 
    1. Finds ANY model with 'flash' in the name (regardless of version).
    2. Strictly avoids any model with 'pro' in the name to prevent 429 errors.
    """
    genai.configure(api_key=api_key)
    try:
        # Get all models that can generate content
        all_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 1. Search for ANY Flash model (gemini-1.5-flash, gemini-2.0-flash, etc.)
        for model_name in all_models:
            if 'flash' in model_name.lower():
                return genai.GenerativeModel(model_name)
        
        # 2. If no Flash model found, search for ANY model that is NOT 'pro'
        for model_name in all_models:
            if 'pro' not in model_name.lower():
                return genai.GenerativeModel(model_name)
        
        # 3. Absolute last resort: pick the first available model if nothing else works
        if all_models:
            return genai.GenerativeModel(all_models[0])
            
    except Exception as e:
        st.error(f"Model selection error: {e}")
    return None

# --- Backend Logic ---
def get_pubmed_data(email, query, max_results=20):
    Entrez.email = email
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        id_list = record["IdList"]
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        docs = []
        for article in records.get('PubmedArticle', []):
            pmid = article['MedlineCitation']['PMID']
            abstract_list = article['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', [])
            if abstract_list:
                docs.append({"text": " ".join([str(p) for p in abstract_list]), "pmid": str(pmid)})
        return docs
    except Exception as e:
        st.error(f"PubMed Error: {e}")
        return []

# --- UI Elements ---
query = st.text_input("Search Query", placeholder="e.g., CRISPR lung cancer oncogene")
claim = st.text_area("Claim to Cite", placeholder="e.g., CRISPR technology can target SOX2 in lung cancer.")

if st.button("Generate Citation"):
    if not api_key or not email or not query or not claim:
        st.error("Please fill in all fields in the sidebar and main page!")
    else:
        try:
            with st.spinner("Initializing AI and searching PubMed..."):
                # 1. Auto-select the correct model
                model = get_best_model(api_key)
                if model is None:
                    st.error("Could not find a compatible Gemini model for this API key.")
                    st.stop()

                # 2. Initialize Embedding Model
                embedder = SentenceTransformer('all-MiniLM-L6-v2')
                
                # 3. Fetch and Index
                docs = get_pubmed_data(email, query)
                if not docs:
                    st.warning("No papers found for this query.")
                else:
                    texts = [d['text'] for d in docs]
                    embeddings = embedder.encode(texts)
                    index = faiss.IndexFlatL2(embeddings.shape[1])
                    index.add(np.array(embeddings).astype('float32'))
                    
                    # 4. Retrieve
                    query_vec = embedder.encode([claim])
                    _, indices = index.search(np.array(query_vec).astype('float32'), 3)
                    relevant = [docs[idx] for idx in indices[0] if idx != -1]
                    
                    context = "\n\n".join([f"PMID: {d['pmid']}\nCONTENT: {d['text']}" for d in relevant])
                    
                    # 5. Generate
                    prompt = f"""
                    You are a precise scientific data extractor.
                    
                    USER CLAIM: "{claim}"
                    
                    RESEARCH DATA:
                    {context}
                    
                    TASK:
                    1. Search the RESEARCH DATA for the exact sentence that provides the evidence for the user claim.
                    2. Copy that sentence WORD-FOR-WORD. 
                    3. Do NOT rewrite the sentence. 
                    4. Do NOT paraphrase. 
                    5. Do NOT summarize.
                    6. Do NOT change any numbers, percentages, or terminology.
                    7. Place the exact sentence in quotation marks " ".
                    8. Append the ACTUAL PMID number in this format: [PMID: 12345678].
                    
                    If no sentence in the provided data directly supports the claim, state "No direct quote found."
                    
                    RESPONSE:
                    """
                    response = model.generate_content(prompt)
                    
                    st.success("Done!")
                    st.markdown("### Result:")
                    st.write(response.text)
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
