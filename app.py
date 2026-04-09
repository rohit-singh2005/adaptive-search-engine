import streamlit as st
import numpy as np
import pandas as pd
from ir_engine import IREngine
from datetime import datetime
import time
import os

# --- Page Config ---
st.set_page_config(page_title="Adaptive Search Engine", layout="wide", page_icon="🔍")

# --- Custom Styling (Modern Indigo Theme) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&family=Space+Grotesk:wght@500;700&display=swap');

    .stApp {
        background-color: #0f0f1a;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 1px;
        color: #6C63FF;
    }

    /* Card Styling */
    .ase-card-box {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(108, 99, 255, 0.15);
        height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.3s ease;
    }
    .ase-card-box:hover {
        transform: translateY(-4px);
        border: 1px solid rgba(108, 99, 255, 0.5);
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
    }
    
    .hero-card {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, rgba(108, 99, 255, 0.15) 100%);
        border-radius: 16px;
        padding: 48px 56px;
        margin-bottom: 40px;
        border-left: 6px solid #6C63FF;
        box-shadow: 0 4px 24px rgba(108, 99, 255, 0.1);
    }
    
    .match-percent {
        color: #00d4aa;
        font-weight: 700;
        font-size: 1.1rem;
    }

    /* Override Streamlit Column Spacing */
    [data-testid="stHorizontalBlock"] {
        gap: 0rem !important;
    }

    /* Feedback Button Styling */
    .stButton>button {
        background-color: transparent !important;
        color: #a3a3c2 !important;
        border: 1px solid #2a2a4a !important;
        border-radius: 8px !important;
        padding: 2px 10px !important;
        font-size: 0.8rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background-color: #6C63FF !important;
        border-color: #6C63FF !important;
        color: white !important;
        box-shadow: 0 2px 12px rgba(108, 99, 255, 0.3) !important;
    }

    /* Profile Button Styling */
    .profile-btn .stButton>button {
        font-size: 2.5rem !important;
        padding: 16px !important;
        border-radius: 16px !important;
        background: #1a1a2e !important;
        border: 2px solid #2a2a4a !important;
    }
    .profile-btn .stButton>button:hover {
        border-color: #6C63FF !important;
        background: rgba(108, 99, 255, 0.1) !important;
        transform: scale(1.05);
    }

    /* Container borders */
    [data-testid="stContainer"] {
        background: #1a1a2e !important;
        border: 1px solid rgba(108, 99, 255, 0.12) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stContainer"]:hover {
        border-color: rgba(108, 99, 255, 0.4) !important;
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.08) !important;
    }

    /* Search input */
    .stTextInput>div>div>input {
        background-color: #1a1a2e !important;
        color: #e0e0e0 !important;
        border: 1px solid #2a2a4a !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-size: 1rem !important;
    }
    .stTextInput>div>div>input:focus {
        border-color: #6C63FF !important;
        box-shadow: 0 0 0 2px rgba(108, 99, 255, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State / Engine Initialization ---
if 'engine' not in st.session_state:
    with st.status("🔍 Setting up Adaptive Search Engine...", expanded=True) as status:
        st.write("Initializing IR Engine...")
        engine = IREngine(subset_size=5000)
        
        st.write("Checking for cached MS MARCO data...")
        is_cached = engine.load_data()
        
        st.write("Ensuring vector index is ready...")
        engine.build_index()
        
        st.session_state.engine = engine
        st.session_state.profile_vec = None
        st.session_state.feedback_trigger = 0
        status.update(label="✅ Engine ready!", state="complete", expanded=False)

# --- Landing Screen: Who's Searching? ---
if st.session_state.profile_vec is None:
    st.write("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #e0e0e0; font-size: 3.5rem;'>Who's searching?</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6a6a8a; font-size: 1.1rem; margin-bottom: 40px;'>Choose a profile to personalize your results</p>", unsafe_allow_html=True)
    
    col_p = st.columns(5)
    profiles = [
        {"name": "Researcher", "emoji": "🧠", "query": "scientific breakthroughs quantum discoveries space exploration"},
        {"name": "Tech Wizard", "emoji": "💻", "query": "latest software engineering artificial intelligence hardware news"},
        {"name": "History Buff", "emoji": "📜", "query": "ancient civilizations world wars historical biographies"},
        {"name": "Health Guru", "emoji": "🧬", "query": "nutrition fitness mental health wellness medical research"},
        {"name": "Explore All", "emoji": "🌐", "query": ""}
    ]
    
    for i, p in enumerate(profiles):
        with col_p[i]:
            st.markdown(f"<div style='text-align: center;'>", unsafe_allow_html=True)
            if st.button(p['emoji'], key=f"avatar_{i}"):
                if p['query']:
                    st.session_state.profile_vec = st.session_state.engine.get_embedding(p['query'])
                else:
                    st.session_state.profile_vec = np.zeros(st.session_state.engine.index.d)
                st.rerun()
            st.markdown(f"<p style='color: #6a6a8a; font-size: 1.1rem; margin-top: 5px;'>{p['name']}</p></div>", unsafe_allow_html=True)
    st.stop()

# --- App Header ---
t_col1, t_col2, t_col3 = st.columns([1, 2, 1])
with t_col1:
    st.markdown("<h1 style='color: #6C63FF; font-size: 2rem; margin-top: 10px;'>🔍 ADAPTIVE SEARCH</h1>", unsafe_allow_html=True)
with t_col2:
    query = st.text_input("", placeholder="Search the MS MARCO database...", label_visibility="collapsed")
with t_col3:
    if st.button("Change Profile 👤"):
        st.session_state.profile_vec = None
        st.rerun()

# --- Results ---
if query:
    results_p = st.session_state.engine.search(query, profile_vec=st.session_state.profile_vec, personalization_weight=0.5)
    results_b = st.session_state.engine.search(query, profile_vec=None, personalization_weight=0.0)

    # 1. Hero Result
    hero = results_p.iloc[0]
    st.markdown(f"""
    <div class="hero-card">
        <h2 style="color: #e0e0e0; font-size: 2.8rem; margin:0;">{hero['title']}</h2>
        <p style="margin: 15px 0;"><span class="match-percent">{int(hero['score']*100)}% Match</span> &nbsp;·&nbsp; MS MARCO &nbsp;·&nbsp; Adaptive Re-ranking</p>
        <p style="font-size: 1.15rem; max-width: 900px; color: #b0b0c8; line-height: 1.7;">{hero['content']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Function to render interactive result grids
    def render_result_row(df, title, key_prefix):
        st.markdown(f"### {title}")
        
        items = df.head(12)
        cols = st.columns(4)
        for i, (idx, row) in enumerate(items.iterrows()):
            with cols[i % 4]:
                with st.container(border=True):
                    st.markdown(f"**{row['title']}**")
                    st.markdown(f"<p style='font-size: 0.8rem; color:#7a7a9a; height: 80px; overflow: hidden;'>{row['content'][:120]}...</p>", unsafe_allow_html=True)
                    
                    st.markdown(f"<span class='match-percent'>{int(row['score']*100)}% Match</span>", unsafe_allow_html=True)
                    
                    btn_col1, btn_col2 = st.columns(2)
                    if btn_col1.button("👍", key=f"{key_prefix}_up_{idx}"):
                        vec = st.session_state.engine.embeddings[row['id']]
                        st.session_state.profile_vec = st.session_state.engine.rocchio_update(st.session_state.profile_vec, [vec], [])
                        st.session_state.feedback_trigger += 1
                        st.rerun()
                    if btn_col2.button("👎", key=f"{key_prefix}_down_{idx}"):
                        vec = st.session_state.engine.embeddings[row['id']]
                        st.session_state.profile_vec = st.session_state.engine.rocchio_update(st.session_state.profile_vec, [], [vec])
                        st.session_state.feedback_trigger += 1
                        st.rerun()

    # 2. Personalized Results
    render_result_row(results_p, "TOP PICKS FOR YOU", "pers")

    st.write("<br>", unsafe_allow_html=True)

    # 3. Global Results
    render_result_row(results_b, "GLOBAL SEARCH RESULTS", "base")

else:
    st.markdown("""
    <div style='text-align: center; margin-top: 120px;'>
        <p style='font-size: 3rem; margin-bottom: 8px;'>🔍</p>
        <h3 style='color: #4a4a6a;'>Search for anything to see personalized results</h3>
        <p style='color: #3a3a5a; font-size: 0.95rem;'>Your results adapt in real-time based on your feedback</p>
    </div>
    """, unsafe_allow_html=True)
