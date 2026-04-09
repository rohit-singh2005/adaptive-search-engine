import streamlit as st
import numpy as np
import pandas as pd
from ir_engine import IREngine
from datetime import datetime
import time
import os

# --- Page Config ---
st.set_page_config(page_title="Netflix Search - MS MARCO", layout="wide", page_icon="🎬")

# --- Custom Styling (Netflix Cinematic Theme) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&family=Bebas+Neue&display=swap');

    .stApp {
        background-color: #141414;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Bebas Neue', cursive;
        letter-spacing: 2px;
        color: #E50914;
    }

    /* Horizontal Scroll Container for Native Columns */
    .stHorizontalScroll {
        overflow-x: auto;
        display: flex;
        flex-direction: row;
        gap: 20px;
        padding: 20px 5px;
        scrollbar-width: thin;
        scrollbar-color: #E50914 #141414;
    }
    
    /* Netflix Card Styling */
    .netflix-card-box {
        background: #181818;
        border-radius: 4px;
        padding: 20px;
        border: 1px solid #2f2f2f;
        height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.3s ease;
    }
    .netflix-card-box:hover {
        transform: scale(1.03);
        border: 1px solid #E50914;
    }
    
    .hero-card {
        background: linear-gradient(90deg, #141414 0%, rgba(20,20,20,0.8) 60%, rgba(229,9,20,0.2) 100%), #181818;
        border-radius: 8px;
        padding: 60px;
        margin-bottom: 40px;
        border-left: 8px solid #E50914;
    }
    
    .match-percent {
        color: #46d369;
        font-weight: bold;
        font-size: 1.1rem;
    }

    /* Override Streamlit Column Spacing */
    [data-testid="stHorizontalBlock"] {
        gap: 0rem !important;
    }

    /* Feedback Button Styling */
    .stButton>button {
        background-color: transparent !important;
        color: white !important;
        border: 1px solid #444 !important;
        padding: 2px 10px !important;
        font-size: 0.8rem !important;
    }
    .stButton>button:hover {
        background-color: #E50914 !important;
        border-color: #E50914 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State / Engine Initializing ---
if 'engine' not in st.session_state:
    with st.status("🎬 Setting up Netflix Search Experience...", expanded=True) as status:
        st.write("Initializing IR Engine...")
        engine = IREngine(subset_size=5000)
        
        st.write("Checking Local 'data/' Directory for Cached MS MARCO...")
        is_cached = engine.load_data()
        
        st.write("Ensuring Vector Index is Ready...")
        engine.build_index()
        
        st.session_state.engine = engine
        st.session_state.profile_vec = None
        st.session_state.feedback_trigger = 0
        status.update(label="✅ Ready!", state="complete", expanded=False)

# --- Landing Screen: Who's Searching? ---
if st.session_state.profile_vec is None:
    st.write("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: white; font-size: 4rem;'>Who's searching?</h1>", unsafe_allow_html=True)
    
    col_p = st.columns(5)
    profiles = [
        {"name": "Researcher", "emoji": "🧠", "query": "scientific breakthroughs quantum discoveries space exploration"},
        {"name": "Tech Wizard", "emoji": "🛰️", "query": "latest software engineering artificial intelligence hardware news"},
        {"name": "History Buff", "emoji": "📜", "query": "ancient civilizations world wars historical biographies"},
        {"name": "Health Guru", "emoji": "🧘", "query": "nutrition fitness mental health wellness medical research"},
        {"name": "Explore All", "emoji": "🌟", "query": ""}
    ]
    
    for i, p in enumerate(profiles):
        with col_p[i]:
            st.markdown(f"<div style='text-align: center;'>", unsafe_allow_html=True)
            if st.button(p['emoji'], key=f"avatar_{i}"):
                if p['query']:
                    st.session_state.profile_vec = st.session_state.engine.get_embedding(p['query'])
                else:
                    # Clear profile for "Explore All"
                    st.session_state.profile_vec = np.zeros(st.session_state.engine.index.d)
                st.rerun()
            st.markdown(f"<p style='color: #a3a3a3; font-size: 1.2rem; margin-top: 5px;'>{p['name']}</p></div>", unsafe_allow_html=True)
    st.stop()

# --- App Header ---
t_col1, t_col2, t_col3 = st.columns([1, 2, 1])
with t_col1:
    st.markdown("<h1 style='color: #E50914; font-size: 2.5rem; margin-top: 10px;'>NETFLIX SEARCH</h1>", unsafe_allow_html=True)
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
        <h2 style="color: white; font-size: 3.5rem; margin:0;">{hero['title']}</h2>
        <p style="margin: 15px 0;"><span class="match-percent">{int(hero['score']*100)}% Match</span> | Collection: MS MARCO | Real-time Adaptive Reranking </p>
        <p style="font-size: 1.3rem; max-width: 900px; color: #e5e5e5; line-height: 1.6;">{hero['content']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Function to render interactive carousels using native components
    def render_netflix_row(df, title, key_prefix):
        st.markdown(f"### {title}")
        
        # We use a container that we will style via CSS to be horizontally scrollable if needed
        # However, for 10-15 results, a wide columns list works well if properly styled
        items = df.head(12)
        
        # Streamlit doesn't support easy horizontal scrolling for native components out of box.
        # We'll use a clean grid layout (2 rows of 6) which feels modern and structured.
        # Alternatively, we could use a single wide row of columns but that often wraps.
        
        cols = st.columns(4) # 4 columns per row for clarity
        for i, (idx, row) in enumerate(items.iterrows()):
            with cols[i % 4]:
                with st.container(border=True):
                    # Card Content
                    st.markdown(f"**{row['title']}**")
                    st.markdown(f"<p style='font-size: 0.8rem; color:#aaa; height: 80px; overflow: hidden;'>{row['content'][:120]}...</p>", unsafe_allow_html=True)
                    
                    # Bottom Action area
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

    # 2. Row: Top Picks for You
    render_netflix_row(results_p, "TOP PICKS FOR YOU", "pers")

    st.write("<br>", unsafe_allow_html=True)

    # 3. Row: Global Results
    render_netflix_row(results_b, "GLOBAL SEARCH RESULTS", "base")

else:
    st.markdown("<h3 style='text-align: center; color: #666; margin-top: 100px;'>Search for anything to see personalized insights...</h3>", unsafe_allow_html=True)
