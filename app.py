import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ir_engine import IREngine
import os
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="Adaptive Search Engine", layout="wide", page_icon="🔍")

# --- Custom Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    .stApp {
        background: linear-gradient(160deg, #0a0a12 0%, #0d1117 40%, #0a0f1a 100%);
        color: #d0d7e2;
        font-family: 'Poppins', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        letter-spacing: -0.3px;
    }

    /* ---- Landing ---- */
    .landing-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 50px 0 6px;
        letter-spacing: -0.5px;
    }
    .landing-accent {
        color: #06b6d4;
    }
    .landing-sub {
        text-align: center;
        color: #4a5568;
        font-size: 0.92rem;
        margin-bottom: 12px;
        font-weight: 400;
        max-width: 520px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }
    .landing-divider {
        width: 40px;
        height: 2px;
        background: #06b6d4;
        margin: 20px auto 36px;
        border-radius: 1px;
    }

    /* Feature highlights */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        max-width: 700px;
        margin: 0 auto 48px;
    }
    .feature-item {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 18px 16px;
        text-align: left;
    }
    .feature-icon {
        font-size: 1.1rem;
        margin-bottom: 8px;
    }
    .feature-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 3px;
    }
    .feature-desc {
        font-size: 0.7rem;
        color: #4a5568;
        line-height: 1.4;
    }

    /* User select section */
    .section-label {
        text-align: center;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #374151;
        margin-bottom: 18px;
        font-family: 'JetBrains Mono', monospace;
    }

    .avatar-card {
        background: rgba(15, 23, 42, 0.35);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 22px 14px 16px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .avatar-card:hover {
        border-color: rgba(6, 182, 212, 0.3);
        background: rgba(15, 23, 42, 0.55);
    }
    .avatar-icon {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        margin: 0 auto 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        font-weight: 700;
        color: #fff;
    }
    .avatar-name {
        font-size: 0.82rem;
        font-weight: 500;
        color: #64748b;
    }

    /* ---- Questionnaire ---- */
    .questionnaire-title {
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 48px 0 4px;
    }
    .questionnaire-sub {
        text-align: center;
        color: #4a5568;
        font-size: 0.9rem;
        margin-bottom: 32px;
    }
    .q-section-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #06b6d4;
        margin-bottom: 6px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ---- Search Header ---- */
    .search-title {
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #06b6d4, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 24px 0 2px;
    }
    .search-subtitle {
        text-align: center;
        color: #374151;
        font-size: 0.85rem;
        margin-bottom: 24px;
    }

    .stTextInput>div>div>input {
        background-color: rgba(15, 23, 42, 0.7) !important;
        color: #d0d7e2 !important;
        border: 1.5px solid rgba(6, 182, 212, 0.2) !important;
        border-radius: 14px !important;
        padding: 14px 20px !important;
        font-size: 1rem !important;
        font-family: 'Poppins', sans-serif !important;
        backdrop-filter: blur(8px);
    }
    .stTextInput>div>div>input:focus {
        border-color: #06b6d4 !important;
        box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.1) !important;
    }

    /* ---- Result Row ---- */
    .result-row {
        background: rgba(15, 23, 42, 0.5);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(6, 182, 212, 0.08);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 10px;
        display: flex;
        align-items: flex-start;
        gap: 18px;
        transition: all 0.25s ease;
    }
    .result-row:hover {
        border-color: rgba(6, 182, 212, 0.25);
        background: rgba(15, 23, 42, 0.7);
        transform: translateX(4px);
    }
    .result-rank {
        color: #1e3a5f;
        font-size: 0.8rem;
        font-weight: 700;
        min-width: 28px;
        padding-top: 2px;
        font-family: 'JetBrains Mono', monospace;
    }
    .result-body { flex: 1; min-width: 0; }
    .result-title {
        font-size: 1.02rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 6px;
        line-height: 1.35;
    }
    .result-snippet {
        font-size: 0.86rem;
        color: #64748b;
        line-height: 1.6;
        margin-bottom: 8px;
    }
    .result-meta {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.78rem;
        flex-wrap: wrap;
    }
    .relevance-tag {
        color: #10b981;
        font-weight: 700;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
    }
    .source-badge {
        background: rgba(6, 182, 212, 0.08);
        color: #64748b;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ---- Hero Result ---- */
    .hero-result {
        background: linear-gradient(135deg, rgba(15,23,42,0.7) 0%, rgba(6,182,212,0.05) 100%);
        border: 1px solid rgba(6, 182, 212, 0.15);
        border-left: 4px solid #06b6d4;
        border-radius: 14px;
        padding: 28px 32px;
        margin: 16px 0 24px;
        backdrop-filter: blur(12px);
    }
    .hero-result h2 {
        color: #f1f5f9;
        font-size: 1.4rem;
        margin: 0 0 10px;
    }
    .hero-result p.content {
        color: #7c8ca3;
        font-size: 0.95rem;
        line-height: 1.7;
        max-width: 860px;
    }

    /* ---- Section Header ---- */
    .section-header {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        color: #06b6d4;
        margin: 28px 0 14px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ---- Explanation Box ---- */
    .explain-box {
        background: rgba(6, 182, 212, 0.04);
        border: 1px solid rgba(6, 182, 212, 0.1);
        border-left: 3px solid #06b6d4;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 4px 0 10px;
        font-size: 0.78rem;
        color: #7c8ca3;
        line-height: 1.5;
    }
    .explain-box .reason { margin: 2px 0; }

    /* ---- Conflict Warning ---- */
    .conflict-banner {
        background: rgba(245, 158, 11, 0.06);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 10px;
        padding: 12px 20px;
        margin: 12px 0 20px;
        color: #d4a036;
        font-size: 0.85rem;
    }

    /* ---- Stats Card ---- */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 16px 0 24px;
    }
    .stat-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(6, 182, 212, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #06b6d4;
    }
    .stat-label {
        font-size: 0.72rem;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 4px;
    }

    /* ---- Buttons ---- */
    .stButton>button {
        background-color: transparent !important;
        color: #64748b !important;
        border: 1px solid rgba(6, 182, 212, 0.15) !important;
        border-radius: 8px !important;
        padding: 1px 8px !important;
        font-size: 0.75rem !important;
        font-family: 'Poppins', sans-serif !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #06b6d4, #10b981) !important;
        border-color: transparent !important;
        color: white !important;
        box-shadow: 0 4px 16px rgba(6, 182, 212, 0.25) !important;
    }

    .show-more-container { text-align: center; margin: 20px 0 40px; }
    [data-testid="stHorizontalBlock"] { gap: 0rem !important; }

    /* ---- Topbar ---- */
    .topbar-badge {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(6, 182, 212, 0.15);
        border-radius: 10px;
        padding: 6px 16px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.85rem;
        color: #94a3b8;
    }
    .topbar-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }

    /* ---- Empty State ---- */
    .empty-state { text-align: center; margin-top: 100px; }
    .empty-state h3 { color: #1e3a5f; font-size: 1.1rem; }
    .empty-state p { color: #1a2a42; font-size: 0.85rem; margin-top: 4px; }

    /* ---- Feedback bar ---- */
    .feedback-bar {
        text-align: center;
        color: #2a4a6a;
        font-size: 0.8rem;
        margin-top: 36px;
        padding: 10px;
        border-top: 1px solid rgba(6, 182, 212, 0.1);
    }

    /* ---- Expansion Banner ---- */
    .expansion-banner {
        background: rgba(16, 185, 129, 0.06);
        border: 1px solid rgba(16, 185, 129, 0.15);
        border-radius: 10px;
        padding: 10px 18px;
        margin: 8px 0 16px;
        font-size: 0.82rem;
        color: #10b981;
    }
    .expansion-banner b { color: #34d399; }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: rgba(10, 15, 26, 0.95) !important;
        border-right: 1px solid rgba(6, 182, 212, 0.08) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- User definitions ---
USER_COLORS = ["#06b6d4", "#10b981", "#8b5cf6", "#f59e0b", "#ec4899"]

# Init editable names
if 'user_names' not in st.session_state:
    st.session_state.user_names = [f"User {i+1}" for i in range(5)]

# --- Questionnaire Options ---
INTEREST_OPTIONS = ["Tech", "Health", "Finance", "Sports", "Entertainment"]
DEPTH_OPTIONS = ["quick", "detailed", "research"]
RECENCY_OPTIONS = ["last week", "last month", "last year", "any"]
SOURCE_OPTIONS = ["official", "blog", "news", "forum"]
AVOID_OPTIONS = ["Politics", "Gossip", "Ads"]

# --- Engine init ---
if 'engine' not in st.session_state:
    with st.status("Initializing engine...", expanded=True) as status:
        st.write("Loading IR Engine...")
        engine = IREngine(subset_size=5000)
        st.write("Loading MS MARCO data...")
        engine.load_data()
        st.write("Building vector index + BM25...")
        engine.build_index()

        st.session_state.engine = engine
        st.session_state.current_user = None
        st.session_state.questionnaire_done = False
        st.session_state.feedback_count = 0
        st.session_state.num_results_pers = 10
        st.session_state.num_results_base = 10
        st.session_state.user_profiles = {}
        st.session_state.user_prefs = {}
        st.session_state.liked_docs = {}
        st.session_state.search_history = {}
        st.session_state.feedback_log = {}
        st.session_state.eval_history = {}
        st.session_state.liked_ids = {}
        st.session_state.disliked_ids = {}

        # Init profiles for default users
        for i in range(5):
            st.session_state.user_profiles[f"User {i+1}"] = np.zeros(engine.index.d)

        # Load persisted profiles if available
        saved = engine.load_profiles()
        if saved:
            profiles, prefs, liked_docs, feedback_log = saved
            st.session_state.user_profiles.update(profiles)
            st.session_state.user_prefs.update(prefs)
            st.session_state.liked_docs.update(liked_docs)
            st.session_state.feedback_log.update(feedback_log)
            for un in prefs:
                st.session_state.search_history.setdefault(un, [])
                st.session_state.eval_history.setdefault(un, [])
                st.session_state.liked_ids.setdefault(un, [])
                st.session_state.disliked_ids.setdefault(un, [])
            st.write("Loaded saved user profiles from disk.")

        status.update(label="Ready", state="complete", expanded=False)

# ================================================================
#  SCREEN 1 — Landing Dashboard
# ================================================================
if st.session_state.current_user is None:
    st.session_state.questionnaire_done = False

    st.markdown("<h1 class='landing-title'>Adaptive <span class='landing-accent'>Search</span> Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p class='landing-sub'>An information retrieval system that learns from your behavior to deliver personalized, explainable search results over the MS MARCO corpus.</p>", unsafe_allow_html=True)
    st.markdown("<div class='landing-divider'></div>", unsafe_allow_html=True)

    # Feature highlights
    st.markdown("""
    <div class='feature-grid'>
        <div class='feature-item'>
            <div class='feature-icon'>⚡</div>
            <div class='feature-label'>Hybrid Retrieval</div>
            <div class='feature-desc'>Dense semantic search + BM25 keyword matching combined for better results</div>
        </div>
        <div class='feature-item'>
            <div class='feature-icon'>🧠</div>
            <div class='feature-label'>Adaptive Profiles</div>
            <div class='feature-desc'>Rocchio feedback learns your interests from thumbs up/down signals</div>
        </div>
        <div class='feature-item'>
            <div class='feature-icon'>📊</div>
            <div class='feature-label'>Evaluation Metrics</div>
            <div class='feature-desc'>Live P@K, MRR, and Recall tracking across feedback rounds</div>
        </div>
        <div class='feature-item'>
            <div class='feature-icon'>🔍</div>
            <div class='feature-label'>Query Expansion</div>
            <div class='feature-desc'>Profile-driven TF-IDF expansion adds related terms automatically</div>
        </div>
        <div class='feature-item'>
            <div class='feature-icon'>💡</div>
            <div class='feature-label'>Explainable Ranking</div>
            <div class='feature-desc'>Every result shows why it was ranked — query match, profile, source</div>
        </div>
        <div class='feature-item'>
            <div class='feature-icon'>💾</div>
            <div class='feature-label'>Persistent Profiles</div>
            <div class='feature-desc'>Preferences and feedback saved to disk across sessions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # User selection at bottom
    st.markdown("<div class='section-label'>Select a profile to begin</div>", unsafe_allow_html=True)

    cols = st.columns([1, 3, 1])
    with cols[1]:
        inner_cols = st.columns(5)
        for i in range(5):
            with inner_cols[i]:
                color = USER_COLORS[i]
                display_name = st.session_state.user_names[i]
                initials = display_name[:2].upper()
                st.markdown(f"""
                <div class='avatar-card'>
                    <div class='avatar-icon' style='background:{color};'>{initials}</div>
                    <div class='avatar-name'>{display_name}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Select", key=f"sel_{i}", use_container_width=True):
                    uname = st.session_state.user_names[i]
                    st.session_state.current_user = {"name": uname, "color": color, "idx": i}
                    if uname not in st.session_state.user_profiles:
                        st.session_state.user_profiles[uname] = np.zeros(st.session_state.engine.index.d)
                    if uname in st.session_state.user_prefs:
                        st.session_state.questionnaire_done = True
                    st.rerun()

    st.stop()

# ================================================================
#  SCREEN 2 — Initial Questionnaire (with editable name)
# ================================================================
active_user = st.session_state.current_user

if not st.session_state.questionnaire_done and active_user["name"] not in st.session_state.user_prefs:
    st.markdown(f"<h1 class='questionnaire-title'>Set up your preferences</h1>", unsafe_allow_html=True)
    st.markdown("<p class='questionnaire-sub'>This helps us personalize your search results from the start</p>", unsafe_allow_html=True)

    _, form_col, _ = st.columns([1, 2.5, 1])
    with form_col:
        # Editable display name
        st.markdown("<div class='q-section-label'>Your Display Name</div>", unsafe_allow_html=True)
        new_name = st.text_input("Display Name", value=active_user["name"],
                                  label_visibility="collapsed", placeholder="Enter your name")

        with st.form("questionnaire_form"):
            st.markdown("<div class='q-section-label'>Topics of Interest</div>", unsafe_allow_html=True)
            interests = st.multiselect(
                "Select your interests",
                INTEREST_OPTIONS,
                default=["Tech"],
                label_visibility="collapsed"
            )

            st.markdown("<div class='q-section-label' style='margin-top:16px;'>Search Depth</div>", unsafe_allow_html=True)
            depth = st.radio(
                "How detailed should results be?",
                DEPTH_OPTIONS,
                horizontal=True,
                label_visibility="collapsed"
            )

            st.markdown("<div class='q-section-label' style='margin-top:16px;'>Recency Preference</div>", unsafe_allow_html=True)
            recency = st.radio(
                "How recent?",
                RECENCY_OPTIONS,
                horizontal=True,
                label_visibility="collapsed"
            )

            st.markdown("<div class='q-section-label' style='margin-top:16px;'>Preferred Source Type</div>", unsafe_allow_html=True)
            source_pref = st.radio(
                "Source type",
                SOURCE_OPTIONS,
                horizontal=True,
                label_visibility="collapsed"
            )

            st.markdown("<div class='q-section-label' style='margin-top:16px;'>Topics to Avoid</div>", unsafe_allow_html=True)
            avoid = st.multiselect(
                "Select topics to de-prioritize",
                AVOID_OPTIONS,
                default=[],
                label_visibility="collapsed"
            )

            col_submit, col_skip = st.columns(2)
            submitted = col_submit.form_submit_button("Apply Preferences", use_container_width=True)
            skipped = col_skip.form_submit_button("Skip", use_container_width=True)

            if submitted:
                # Handle name change
                old_name = active_user["name"]
                final_name = new_name.strip() if new_name.strip() else old_name
                if final_name != old_name:
                    idx = active_user.get("idx", 0)
                    st.session_state.user_names[idx] = final_name
                    if old_name in st.session_state.user_profiles:
                        st.session_state.user_profiles[final_name] = st.session_state.user_profiles.pop(old_name)
                    st.session_state.current_user = {"name": final_name, "color": active_user["color"], "idx": idx}

                uname = final_name
                prefs = {
                    "interests": interests,
                    "depth": depth,
                    "recency": recency,
                    "source_pref": source_pref,
                    "avoid": avoid,
                }
                st.session_state.user_prefs[uname] = prefs
                st.session_state.liked_docs[uname] = []
                st.session_state.search_history.setdefault(uname, [])
                st.session_state.feedback_log.setdefault(uname, [])
                st.session_state.eval_history.setdefault(uname, [])
                st.session_state.liked_ids.setdefault(uname, [])
                st.session_state.disliked_ids.setdefault(uname, [])

                # Build initial profile vector from questionnaire
                initial_vec = st.session_state.engine.build_initial_profile(interests, depth, source_pref)
                st.session_state.user_profiles[uname] = initial_vec

                # Persist to disk
                st.session_state.engine.save_profiles(
                    st.session_state.user_profiles, st.session_state.user_prefs,
                    st.session_state.liked_docs, st.session_state.feedback_log)

                st.session_state.questionnaire_done = True
                st.rerun()

            if skipped:
                old_name = active_user["name"]
                final_name = new_name.strip() if new_name.strip() else old_name
                if final_name != old_name:
                    idx = active_user.get("idx", 0)
                    st.session_state.user_names[idx] = final_name
                    if old_name in st.session_state.user_profiles:
                        st.session_state.user_profiles[final_name] = st.session_state.user_profiles.pop(old_name)
                    st.session_state.current_user = {"name": final_name, "color": active_user["color"], "idx": idx}

                uname = final_name
                st.session_state.user_prefs[uname] = {
                    "interests": [], "depth": "detailed", "recency": "any",
                    "source_pref": "official", "avoid": []
                }
                st.session_state.liked_docs[uname] = []
                st.session_state.search_history.setdefault(uname, [])
                st.session_state.feedback_log.setdefault(uname, [])
                st.session_state.eval_history.setdefault(uname, [])
                st.session_state.liked_ids.setdefault(uname, [])
                st.session_state.disliked_ids.setdefault(uname, [])
                st.session_state.questionnaire_done = True
                st.rerun()

    st.stop()

# ================================================================
#  SCREEN 3 — Search Interface
# ================================================================
uname = active_user["name"]
profile_vec = st.session_state.user_profiles[uname]
user_prefs = st.session_state.user_prefs.get(uname, {})
liked_docs = st.session_state.liked_docs.get(uname, [])
st.session_state.search_history.setdefault(uname, [])
st.session_state.feedback_log.setdefault(uname, [])
st.session_state.eval_history.setdefault(uname, [])
st.session_state.liked_ids.setdefault(uname, [])
st.session_state.disliked_ids.setdefault(uname, [])

# --- Sidebar: Controls + Analytics ---
with st.sidebar:
    st.markdown(f"<div class='topbar-badge'><span class='topbar-dot' style='background:{active_user['color']};'></span> {uname}</div>", unsafe_allow_html=True)
    if st.button("Switch User", use_container_width=True):
        st.session_state.current_user = None
        st.rerun()

    st.markdown("---")
    st.markdown("##### ⚙️ Retrieval Controls")
    pers_weight = st.slider("Personalization Weight", 0.0, 1.0, 0.5, 0.05, help="How much to bias results toward your profile")
    dense_w = st.slider("Dense vs BM25 Blend", 0.0, 1.0, 0.7, 0.05, help="1.0 = pure dense, 0.0 = pure BM25")
    use_expansion = st.toggle("Query Expansion", value=False, help="Expand query using profile-derived terms")

    st.markdown("---")
    st.markdown("##### 📊 Profile Analytics")
    prof_norm = float(np.linalg.norm(profile_vec))
    strength = min(prof_norm * 100, 100)
    st.progress(strength / 100, text=f"Profile Strength: {strength:.0f}%")

    # Topic affinity from profile
    if prof_norm > 0:
        from ir_engine import TOPIC_KEYWORDS
        topic_scores = {}
        for topic, kws in TOPIC_KEYWORDS.items():
            topic_text = " ".join(kws)
            t_vec = st.session_state.engine.get_embedding(topic_text)
            topic_scores[topic] = float(np.dot(profile_vec, t_vec))
        fig = go.Figure(go.Bar(
            x=list(topic_scores.values()), y=list(topic_scores.keys()),
            orientation='h', marker_color='#5046e5'
        ))
        fig.update_layout(height=220, margin=dict(l=0,r=0,t=20,b=0),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='#8888a4', size=11),
                          xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    total_likes = len(st.session_state.liked_ids.get(uname, []))
    total_dislikes = len(st.session_state.disliked_ids.get(uname, []))
    fb_col1, fb_col2 = st.columns(2)
    fb_col1.metric("👍 Likes", total_likes)
    fb_col2.metric("👎 Dislikes", total_dislikes)

    st.markdown("---")
    st.markdown("##### 🕑 Recent Searches")
    history = st.session_state.search_history.get(uname, [])
    if history:
        for h in history[-5:][::-1]:
            st.caption(f"🔍 {h['query'][:40]}...")
    else:
        st.caption("No searches yet")

# --- Search area ---
st.markdown("<div class='search-title'>Adaptive Search Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='search-subtitle'>Results personalize as you provide relevance feedback</div>", unsafe_allow_html=True)

_, center_col, _ = st.columns([1, 3, 1])
with center_col:
    query = st.text_input("", placeholder="Search MS MARCO collection...", label_visibility="collapsed")

# --- Results ---
if query:
    has_profile = np.linalg.norm(profile_vec) > 0
    avoid_topics = user_prefs.get("avoid", [])
    preferred_sources = [user_prefs.get("source_pref", "")] if user_prefs.get("source_pref") else None
    recency_pref = user_prefs.get("recency", "any")

    # Topic mismatch detection
    has_conflict, default_pw = st.session_state.engine.detect_topic_conflict(query, avoid_topics)
    p_weight = min(pers_weight, default_pw) if has_conflict else pers_weight
    if not has_profile:
        p_weight = 0.0

    # Recency weight based on preference
    recency_weight_map = {"last week": 0.15, "last month": 0.10, "last year": 0.05, "any": 0.0}
    r_weight = recency_weight_map.get(recency_pref, 0.0)
    s_weight = 0.1 if preferred_sources else 0.0

    # Conflict banner
    if has_conflict:
        conflict_topic = [t for t in st.session_state.engine.classify_text(query) if t in avoid_topics]
        topic_str = ", ".join(conflict_topic)
        st.markdown(f"""
        <div class='conflict-banner'>
            Personalization reduced — your query touches a topic you prefer to avoid ({topic_str}).
            Showing more neutral results.
        </div>
        """, unsafe_allow_html=True)

    # Execute hybrid search
    results_p, expansion_terms = st.session_state.engine.search(
        query, profile_vec=profile_vec, personalization_weight=p_weight,
        preferred_sources=preferred_sources, recency_weight=r_weight, source_weight=s_weight,
        dense_weight=dense_w, use_bm25=True, expand=use_expansion
    )
    results_b, _ = st.session_state.engine.search(
        query, profile_vec=None, personalization_weight=0.0,
        dense_weight=dense_w, use_bm25=True
    )

    # Show expansion terms if used
    if expansion_terms:
        terms_str = ", ".join(expansion_terms[:5])
        st.markdown(f"<div class='expansion-banner'><b>🔄 Query expanded with:</b> {terms_str}</div>", unsafe_allow_html=True)

    # --- Visible Stats Grid (main area) ---
    prof_str = min(float(np.linalg.norm(profile_vec)) * 100, 100)
    total_fb = len(st.session_state.liked_ids.get(uname, [])) + len(st.session_state.disliked_ids.get(uname, []))
    mode_label = "Hybrid" if dense_w < 1.0 else "Dense Only"
    st.markdown(f"""
    <div class='stats-grid'>
        <div class='stat-card'>
            <div class='stat-value'>{mode_label}</div>
            <div class='stat-label'>Retrieval Mode</div>
        </div>
        <div class='stat-card'>
            <div class='stat-value'>{int(dense_w*100)}:{int((1-dense_w)*100)}</div>
            <div class='stat-label'>Dense : BM25</div>
        </div>
        <div class='stat-card'>
            <div class='stat-value'>{prof_str:.0f}%</div>
            <div class='stat-label'>Profile Strength</div>
        </div>
        <div class='stat-card'>
            <div class='stat-value'>{total_fb}</div>
            <div class='stat-label'>Feedback Signals</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Log search to history
    query_topics = st.session_state.engine.classify_text(query)
    hist_entry = {"query": query, "time": datetime.now().isoformat(), "topics": query_topics[:3]}
    user_hist = st.session_state.search_history.get(uname, [])
    if not user_hist or user_hist[-1]["query"] != query:
        st.session_state.search_history.setdefault(uname, []).append(hist_entry)

    # --- Hero ---
    hero = results_p.iloc[0]
    hero_source = hero.get('source_type', 'general')
    hero_dense = int(hero.get('dense_sim', 0) * 100)
    hero_bm25 = int(hero.get('bm25_sim', 0) * 100)
    st.markdown(f"""
    <div class='hero-result'>
        <h2>{hero['title']}</h2>
        <div class='result-meta' style='margin-bottom:12px;'>
            <span class='relevance-tag'>{int(hero['score']*100)}% relevance</span>
            <span class='source-badge'>{hero_source}</span>
            <span class='source-badge'>Dense: {hero_dense}%</span>
            <span class='source-badge'>BM25: {hero_bm25}%</span>
            <span class='source-badge'>{'✨ Personalized' if has_profile else 'Baseline'}</span>
        </div>
        <p class='content'>{hero['content']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Explain hero result
    hero_reasons = st.session_state.engine.explain_result(hero, query, liked_docs)
    reasons_html = "".join(f"<div class='reason'>· {r}</div>" for r in hero_reasons)
    st.markdown(f"<div class='explain-box'>{reasons_html}</div>", unsafe_allow_html=True)

    # --- Row renderer ---
    def render_result_list(df, section_title, key_prefix):
        st.markdown(f"<div class='section-header'>{section_title}</div>", unsafe_allow_html=True)

        limit_key = f"num_results_{key_prefix}"
        if limit_key not in st.session_state:
            st.session_state[limit_key] = 10

        limit = st.session_state[limit_key]
        items = df.head(limit)

        for rank, (idx, row) in enumerate(items.iterrows(), 1):
            source_type = row.get('source_type', 'general')
            dense_pct = int(row.get('dense_sim', 0) * 100)
            bm25_pct = int(row.get('bm25_sim', 0) * 100)
            st.markdown(f"""
            <div class='result-row'>
                <div class='result-rank'>{rank:02d}</div>
                <div class='result-body'>
                    <div class='result-title'>{row['title']}</div>
                    <div class='result-snippet'>{row['content'][:200]}...</div>
                    <div class='result-meta'>
                        <span class='relevance-tag'>{int(row['score']*100)}%</span>
                        <span class='source-badge'>{source_type}</span>
                        <span class='source-badge'>Dense: {dense_pct}%</span>
                        <span class='source-badge'>BM25: {bm25_pct}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Explainability (for personalized results)
            if key_prefix == "pers":
                reasons = st.session_state.engine.explain_result(row, query, liked_docs)
                reasons_html = "".join(f"<div class='reason'>· {r}</div>" for r in reasons)
                st.markdown(f"<div class='explain-box'>{reasons_html}</div>", unsafe_allow_html=True)

            # Feedback buttons
            _, fb_col, _ = st.columns([6, 1.2, 6])
            with fb_col:
                b1, b2 = st.columns(2)
                if b1.button(":material/thumb_up:", key=f"{key_prefix}_up_{idx}"):
                    vec = st.session_state.engine.embeddings[row['id']]
                    st.session_state.user_profiles[uname] = st.session_state.engine.rocchio_update(
                        st.session_state.user_profiles[uname], [vec], []
                    )
                    st.session_state.liked_docs.setdefault(uname, []).append(row['content'])
                    st.session_state.liked_ids.setdefault(uname, []).append(int(row['id']))
                    st.session_state.feedback_log.setdefault(uname, []).append(
                        {"action": "like", "doc_id": int(row['id']), "time": datetime.now().isoformat()})
                    st.session_state.feedback_count += 1
                    st.session_state.engine.save_profiles(
                        st.session_state.user_profiles, st.session_state.user_prefs,
                        st.session_state.liked_docs, st.session_state.feedback_log)
                    st.rerun()
                if b2.button(":material/thumb_down:", key=f"{key_prefix}_dn_{idx}"):
                    vec = st.session_state.engine.embeddings[row['id']]
                    st.session_state.user_profiles[uname] = st.session_state.engine.rocchio_update(
                        st.session_state.user_profiles[uname], [], [vec]
                    )
                    st.session_state.disliked_ids.setdefault(uname, []).append(int(row['id']))
                    st.session_state.feedback_log.setdefault(uname, []).append(
                        {"action": "dislike", "doc_id": int(row['id']), "time": datetime.now().isoformat()})
                    st.session_state.feedback_count += 1
                    st.session_state.engine.save_profiles(
                        st.session_state.user_profiles, st.session_state.user_prefs,
                        st.session_state.liked_docs, st.session_state.feedback_log)
                    st.rerun()

        # Show More
        if len(df) > limit:
            _, more_col, _ = st.columns([5, 2, 5])
            if more_col.button("Show More Results", key=f"more_{key_prefix}", use_container_width=True):
                st.session_state[limit_key] += 10
                st.rerun()

    # Show personalized results only if user has a profile
    if has_profile:
        render_result_list(results_p, "Adapted for you", "pers")

    render_result_list(results_b, "All results", "base")

    # --- Evaluation Metrics ---
    liked_set = set(st.session_state.liked_ids.get(uname, []))
    if liked_set:
        with st.expander("📈 Retrieval Evaluation Metrics"):
            ranked_ids = results_p['id'].tolist()
            p5 = st.session_state.engine.compute_precision_at_k(ranked_ids, liked_set, k=5)
            p10 = st.session_state.engine.compute_precision_at_k(ranked_ids, liked_set, k=10)
            mrr = st.session_state.engine.compute_mrr(ranked_ids, liked_set)
            recall10 = st.session_state.engine.compute_recall_at_k(ranked_ids, liked_set, k=10)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("P@5", f"{p5:.2f}")
            m2.metric("P@10", f"{p10:.2f}")
            m3.metric("MRR", f"{mrr:.2f}")
            m4.metric("Recall@10", f"{recall10:.2f}")

            # Track eval over time
            st.session_state.eval_history.setdefault(uname, []).append(
                {"P@5": p5, "P@10": p10, "MRR": mrr, "fb_count": st.session_state.feedback_count})
            eval_hist = st.session_state.eval_history[uname]
            if len(eval_hist) > 1:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(y=[e["P@5"] for e in eval_hist], name="P@5",
                                          mode='lines+markers', line=dict(color='#5046e5')))
                fig2.add_trace(go.Scatter(y=[e["MRR"] for e in eval_hist], name="MRR",
                                          mode='lines+markers', line=dict(color='#00c896')))
                fig2.update_layout(height=250, title="Metrics Over Feedback Rounds",
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                   font=dict(color='#8888a4'), xaxis_title="Search Round",
                                   xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#1a1a2e'))
                st.plotly_chart(fig2, use_container_width=True)

    # Feedback count
    if st.session_state.feedback_count > 0:
        st.markdown(f"<div class='feedback-bar'>{st.session_state.feedback_count} feedback signal(s) recorded for {uname}</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class='empty-state'>
        <h3>Enter a query above to begin searching</h3>
        <p>Mark results as relevant or not to personalize future rankings</p>
    </div>
    """, unsafe_allow_html=True)
