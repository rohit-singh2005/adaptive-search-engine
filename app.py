import streamlit as st
import numpy as np
import pandas as pd
from ir_engine import IREngine
import time
import os

# --- Page Config ---
st.set_page_config(page_title="Adaptive Search Engine", layout="wide", page_icon="🔍")

# --- Custom Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

    .stApp {
        background: #0a0a12;
        color: #d0d0e0;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 0.5px;
    }

    /* ---------- User Selection Screen ---------- */
    .user-select-title {
        text-align: center;
        font-size: 2.8rem;
        color: #ffffff;
        margin-top: 60px;
        font-family: 'Space Grotesk', sans-serif;
    }
    .user-select-sub {
        text-align: center;
        color: #5a5a7a;
        font-size: 1.05rem;
        margin-bottom: 48px;
    }

    .user-avatar-card {
        background: #12121f;
        border: 2px solid #1e1e35;
        border-radius: 16px;
        padding: 28px 16px 20px;
        text-align: center;
        transition: all 0.25s ease;
        cursor: pointer;
    }
    .user-avatar-card:hover {
        border-color: #6C63FF;
        box-shadow: 0 0 24px rgba(108, 99, 255, 0.15);
        transform: translateY(-4px);
    }
    .user-avatar-emoji {
        font-size: 3rem;
        margin-bottom: 8px;
    }
    .user-avatar-name {
        font-size: 1rem;
        font-weight: 600;
        color: #c0c0d8;
    }

    /* ---------- Search Bar Area ---------- */
    .search-header {
        text-align: center;
        padding: 48px 0 8px;
    }
    .search-header h1 {
        font-size: 2.2rem;
        color: #6C63FF;
        margin-bottom: 0;
    }
    .search-header p {
        color: #4a4a6a;
        font-size: 0.95rem;
        margin-top: 4px;
    }

    .stTextInput>div>div>input {
        background-color: #12121f !important;
        color: #d0d0e0 !important;
        border: 2px solid #1e1e35 !important;
        border-radius: 14px !important;
        padding: 14px 20px !important;
        font-size: 1.05rem !important;
        transition: border-color 0.2s ease !important;
    }
    .stTextInput>div>div>input:focus {
        border-color: #6C63FF !important;
        box-shadow: 0 0 0 3px rgba(108, 99, 255, 0.15) !important;
    }

    /* ---------- Results ---------- */
    .hero-card {
        background: linear-gradient(135deg, #12121f 0%, #16162a 60%, rgba(108, 99, 255, 0.08) 100%);
        border-radius: 16px;
        padding: 40px 48px;
        margin: 24px 0 32px;
        border-left: 5px solid #6C63FF;
    }
    .hero-card h2 {
        color: #e8e8f0;
        font-size: 1.8rem;
        margin: 0 0 8px;
    }
    .hero-card .hero-meta {
        margin: 8px 0 14px;
    }

    .match-percent {
        color: #00d4aa;
        font-weight: 700;
        font-size: 1rem;
    }

    /* Result cards */
    [data-testid="stContainer"] {
        background: #12121f !important;
        border: 1px solid #1e1e35 !important;
        border-radius: 12px !important;
        transition: all 0.25s ease !important;
    }
    [data-testid="stContainer"]:hover {
        border-color: rgba(108, 99, 255, 0.35) !important;
        box-shadow: 0 4px 16px rgba(108, 99, 255, 0.08) !important;
    }

    [data-testid="stHorizontalBlock"] {
        gap: 0rem !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: transparent !important;
        color: #7a7a9a !important;
        border: 1px solid #1e1e35 !important;
        border-radius: 8px !important;
        padding: 2px 10px !important;
        font-size: 0.8rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background-color: #6C63FF !important;
        border-color: #6C63FF !important;
        color: white !important;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        margin-top: 100px;
    }
    .empty-state .icon { font-size: 3.5rem; margin-bottom: 12px; }
    .empty-state h3 { color: #3a3a58; }
    .empty-state p { color: #2e2e48; font-size: 0.9rem; }

    /* Top bar user badge */
    .user-badge {
        background: #12121f;
        border: 1px solid #1e1e35;
        border-radius: 10px;
        padding: 6px 14px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.9rem;
        color: #a0a0c0;
    }
    .user-badge .badge-avatar { font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# --- User Definitions ---
USERS = [
    {"name": "User 1", "avatar": "👤"},
    {"name": "User 2", "avatar": "👩"},
    {"name": "User 3", "avatar": "👨"},
    {"name": "User 4", "avatar": "🧑"},
    {"name": "User 5", "avatar": "👩‍💻"},
]

# --- Session State / Engine Initialization ---
if 'engine' not in st.session_state:
    with st.status("🔍 Initializing Adaptive Search Engine...", expanded=True) as status:
        st.write("Loading IR Engine...")
        engine = IREngine(subset_size=5000)

        st.write("Checking for cached MS MARCO data...")
        engine.load_data()

        st.write("Building vector index...")
        engine.build_index()

        st.session_state.engine = engine
        st.session_state.current_user = None
        st.session_state.feedback_count = 0

        # Per-user profile vectors (persist across searches within session)
        st.session_state.user_profiles = {
            u["name"]: np.zeros(engine.index.d) for u in USERS
        }
        status.update(label="✅ Engine ready!", state="complete", expanded=False)

# ================================================================
#  SCREEN 1 — User Selection
# ================================================================
if st.session_state.current_user is None:
    st.markdown("<h1 class='user-select-title'>Adaptive Search Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p class='user-select-sub'>A retrieval system that adapts search results based on user behavior and interests</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#3a3a58; margin-bottom:40px; font-size:0.95rem;'>Select a user to begin — each profile learns independently from your feedback</p>", unsafe_allow_html=True)

    cols = st.columns([1, 1, 1, 1, 1])
    for i, user in enumerate(USERS):
        with cols[i]:
            st.markdown(f"""
            <div class='user-avatar-card'>
                <div class='user-avatar-emoji'>{user['avatar']}</div>
                <div class='user-avatar-name'>{user['name']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Select", key=f"sel_{i}", use_container_width=True):
                st.session_state.current_user = user
                st.rerun()

    st.stop()

# ================================================================
#  SCREEN 2 — Search Interface
# ================================================================
active_user = st.session_state.current_user
profile_vec = st.session_state.user_profiles[active_user["name"]]

# --- Top Bar ---
top1, top2, top3 = st.columns([1, 3, 1])
with top1:
    st.markdown(f"""
    <div class='user-badge'>
        <span class='badge-avatar'>{active_user['avatar']}</span>
        <span>{active_user['name']}</span>
    </div>
    """, unsafe_allow_html=True)
with top3:
    if st.button("Switch User 🔄"):
        st.session_state.current_user = None
        st.rerun()

# --- Search Header ---
st.markdown("""
<div class='search-header'>
    <h1>🔍 Adaptive Search Engine</h1>
    <p>Results adapt in real-time based on your feedback</p>
</div>
""", unsafe_allow_html=True)

# --- Search Input ---
_, center_col, _ = st.columns([1, 3, 1])
with center_col:
    query = st.text_input("", placeholder="Search the MS MARCO collection...", label_visibility="collapsed")

# --- Results ---
if query:
    has_profile = np.linalg.norm(profile_vec) > 0
    p_weight = 0.5 if has_profile else 0.0

    results_p = st.session_state.engine.search(query, profile_vec=profile_vec, personalization_weight=p_weight)
    results_b = st.session_state.engine.search(query, profile_vec=None, personalization_weight=0.0)

    # --- Hero Result ---
    hero = results_p.iloc[0]
    st.markdown(f"""
    <div class="hero-card">
        <h2>{hero['title']}</h2>
        <div class="hero-meta">
            <span class="match-percent">{int(hero['score']*100)}% Relevance</span>
            &nbsp;·&nbsp; MS MARCO &nbsp;·&nbsp; {'Personalized' if has_profile else 'Baseline'} Ranking
        </div>
        <p style="font-size: 1.05rem; color: #9a9ab8; line-height: 1.7; max-width: 900px;">{hero['content']}</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Result Grid Renderer ---
    def render_results(df, title, key_prefix):
        st.markdown(f"### {title}")
        items = df.head(12)
        cols = st.columns(4)
        for i, (idx, row) in enumerate(items.iterrows()):
            with cols[i % 4]:
                with st.container(border=True):
                    st.markdown(f"**{row['title']}**")
                    st.markdown(f"<p style='font-size:0.8rem; color:#6a6a88; height:80px; overflow:hidden;'>{row['content'][:130]}...</p>", unsafe_allow_html=True)

                    st.markdown(f"<span class='match-percent'>{int(row['score']*100)}%</span>", unsafe_allow_html=True)

                    b1, b2 = st.columns(2)
                    if b1.button("👍", key=f"{key_prefix}_up_{idx}"):
                        vec = st.session_state.engine.embeddings[row['id']]
                        st.session_state.user_profiles[active_user["name"]] = st.session_state.engine.rocchio_update(
                            st.session_state.user_profiles[active_user["name"]], [vec], []
                        )
                        st.session_state.feedback_count += 1
                        st.rerun()
                    if b2.button("👎", key=f"{key_prefix}_dn_{idx}"):
                        vec = st.session_state.engine.embeddings[row['id']]
                        st.session_state.user_profiles[active_user["name"]] = st.session_state.engine.rocchio_update(
                            st.session_state.user_profiles[active_user["name"]], [], [vec]
                        )
                        st.session_state.feedback_count += 1
                        st.rerun()

    # Personalized row (only show if user has given feedback)
    if has_profile:
        render_results(results_p, "ADAPTED FOR YOU", "pers")
        st.write("<br>", unsafe_allow_html=True)

    render_results(results_b, "ALL RESULTS", "base")

    # --- Feedback indicator ---
    if st.session_state.feedback_count > 0:
        st.markdown(f"<p style='text-align:center; color:#3a3a58; font-size:0.85rem; margin-top:32px;'>📊 {active_user['name']} has given {st.session_state.feedback_count} feedback signal(s) — results are adapting</p>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class='empty-state'>
        <div class='icon'>🔎</div>
        <h3>Enter a query to search</h3>
        <p>Your results will adapt as you provide feedback with 👍 and 👎</p>
    </div>
    """, unsafe_allow_html=True)
