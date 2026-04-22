"""
frontend/app.py  (ExamRAG v5 — powered by Viyaan AI)
──────────────────────────────────────────────────────
"Train Your Mind. Master Your Exams."

v5 new features:
  1. Gamification   — streak tracker, XP points, level badges
  2. Learning intelligence — mode usage tracking, adaptive suggestions
  3. Adaptive difficulty   — beginner → intermediate → advanced prompting
  4. Clean streaming       — single write path, zero double-render
  5. Smart suggestions     — contextual "Try Quiz" / "Revise this" hints
  6. Fully mobile-responsive layout

Bug-free architecture:
  - st.chat_input fires ONCE per submit — no rerun loops
  - st.rerun() only called after state mutation
  - All HTML is static CSS — no user content injected into unsafe strings
  - Backend loaded once via @st.cache_resource

Run:
    streamlit run frontend/app.py
"""

import sys
import os
import time
import json
import datetime
import random
import re as _re

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

# ── Page config — must be FIRST Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="ExamRAG · Viyaan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  THEME / CSS — static only, no Python values injected
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:      #07090e;
  --panel:   #0f1623;
  --panel2:  #141d2b;
  --border:  #1c2840;
  --b2:      #253350;
  --blue:    #5b9cf6;
  --green:   #34d399;
  --amber:   #fbbf24;
  --red:     #f87171;
  --purple:  #a78bfa;
  --orange:  #fb923c;
  --txt:     #dde6f0;
  --txt2:    #7a90a8;
  --txt3:    #3a4f65;
  --sans:    'Inter', sans-serif;
  --mono:    'JetBrains Mono', monospace;
  --r:       10px;
}

html, body, [class*="css"]  { font-family: var(--sans); }
.stApp                      { background: var(--bg) !important; color: var(--txt); }
.block-container            { padding-top: 0.5rem !important; max-width: 1080px; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--panel) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] label { color: var(--txt2) !important; }

div[data-baseweb="select"] > div { background: var(--panel2) !important; border-color: var(--b2) !important; }
div[data-baseweb="select"] span  { color: var(--txt) !important; }

/* Buttons */
.stButton > button {
  background: var(--panel2) !important; color: var(--txt) !important;
  border: 1px solid var(--b2) !important; border-radius: var(--r) !important;
  font-size: .8rem !important; font-weight: 500 !important;
  padding: .35rem .8rem !important; transition: border-color .15s, color .15s !important;
}
.stButton > button:hover { border-color: var(--blue) !important; color: var(--blue) !important; }

/* Chat input */
div[data-testid="stChatInput"] > div {
  background: var(--panel2) !important; border: 1px solid var(--b2) !important;
  border-radius: 12px !important;
}
div[data-testid="stChatInput"] textarea  { color: var(--txt) !important; font-family: var(--sans) !important; font-size: .9rem !important; }
div[data-testid="stChatInput"] button    { background: var(--blue) !important; border-radius: 8px !important; }

/* Chat messages */
div[data-testid="stChatMessage"] { background: transparent !important; padding: 0 !important; }

/* Expander */
details > summary              { font-size: .8rem !important; color: var(--txt2) !important; }
div[data-testid="stExpander"]  { background: var(--panel2) !important; border: 1px solid var(--border) !important; border-radius: var(--r) !important; }

/* File uploader */
div[data-testid="stFileUploader"] { background: var(--panel2); border: 1px dashed var(--b2); border-radius: var(--r); padding: .5rem; }

/* Alert boxes */
div[data-testid="stInfo"]    { background: #0a1c34 !important; border-color: #1d3461 !important; }
div[data-testid="stSuccess"] { background: #051812 !important; }
div[data-testid="stWarning"] { background: #1a1100 !important; }
div[data-testid="stError"]   { background: #1a0808 !important; }

/* Metrics */
div[data-testid="stMetric"]        { background: var(--panel2); border: 1px solid var(--border); border-radius: var(--r); padding: 8px 12px; }
div[data-testid="stMetricValue"]   { color: var(--txt) !important; font-size: 1.1rem !important; }
div[data-testid="stMetricLabel"]   { color: var(--txt3) !important; font-size: .68rem !important; }

/* Progress */
div[data-testid="stProgressBar"] > div        { background: var(--panel) !important; }
div[data-testid="stProgressBar"] > div > div  { background: var(--blue) !important; }

hr { border-color: var(--border) !important; margin: .7rem 0 !important; }

/* Status pills */
.spill     { display:inline-flex; align-items:center; gap:5px; padding:3px 10px; border-radius:20px; font-size:.72rem; font-weight:600; }
.spill-on  { background:rgba(52,211,153,.1);  border:1px solid rgba(52,211,153,.3);  color:#34d399; }
.spill-off { background:rgba(248,113,113,.1); border:1px solid rgba(248,113,113,.3); color:#f87171; }
.sdot      { width:6px; height:6px; border-radius:50%; display:inline-block; }
.sdot-on   { background:#34d399; animation:blink 2s infinite; }
.sdot-off  { background:#f87171; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* Mode / flag tags */
.mtag { display:inline-block; background:#0d1e38; color:#5b9cf6;  border:1px solid rgba(91,156,246,.25); border-radius:5px; padding:1px 8px; font-family:var(--mono); font-size:.63rem; font-weight:600; letter-spacing:.03em; }
.ftag { display:inline-block; background:rgba(251,191,36,.07); color:#fbbf24; border:1px solid rgba(251,191,36,.25); border-radius:5px; padding:1px 8px; font-family:var(--mono); font-size:.63rem; font-weight:600; margin-left:5px; }
.atag { display:inline-block; background:rgba(52,211,153,.07);  color:#34d399; border:1px solid rgba(52,211,153,.25); border-radius:5px; padding:1px 8px; font-family:var(--mono); font-size:.63rem; font-weight:600; margin-left:5px; }

/* Source chip */
.schip { display:inline-block; background:#0a1829; border:1px solid var(--b2); border-radius:5px; padding:2px 9px; font-family:var(--mono); font-size:.63rem; color:var(--txt2); margin:2px; }

/* Exam tip */
.tipbox { background:rgba(251,191,36,.06); border-left:3px solid #fbbf24; border-radius:0 6px 6px 0; padding:8px 13px; margin-top:10px; font-size:.84rem; color:#fbbf24; }

/* Gamification badge */
.xpbadge { display:inline-flex; align-items:center; gap:6px; background:#0d1e38; border:1px solid rgba(91,156,246,.3); border-radius:20px; padding:4px 12px; font-size:.75rem; font-weight:700; color:#5b9cf6; }
.streakbadge { display:inline-flex; align-items:center; gap:6px; background:rgba(251,191,36,.08); border:1px solid rgba(251,191,36,.3); border-radius:20px; padding:4px 12px; font-size:.75rem; font-weight:700; color:#fbbf24; }
.lvlbadge { display:inline-flex; align-items:center; gap:4px; border-radius:20px; padding:4px 12px; font-size:.75rem; font-weight:700; }
.lvl-beginner    { background:rgba(52,211,153,.1);  border:1px solid rgba(52,211,153,.3);  color:#34d399; }
.lvl-intermediate{ background:rgba(91,156,246,.1);  border:1px solid rgba(91,156,246,.3);  color:#5b9cf6; }
.lvl-advanced    { background:rgba(167,139,250,.1); border:1px solid rgba(167,139,250,.3); color:#a78bfa; }

/* Suggestion box */
.suggest { background:rgba(91,156,246,.06); border:1px solid rgba(91,156,246,.2); border-radius:8px; padding:10px 14px; font-size:.82rem; color:#8ab4f8; margin-top:8px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND — loaded once per server process
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _load_backend():
    try:
        from backend.rag_pipeline import (
            query          as _query,
            query_stream   as _stream,
            ingest_bytes,
            get_index_stats,
            get_subjects,
            get_models,
            ollama_online,
            detect_intent,
            MODES,
            DEFAULT_MODEL,
        )
        return dict(
            ok=True,
            query=_query,
            stream=_stream,
            ingest=ingest_bytes,
            stats=get_index_stats,
            subjects=get_subjects,
            models=get_models,
            online=ollama_online,
            detect=detect_intent,
            modes=MODES,
            default_model=DEFAULT_MODEL,
        )
    except Exception as exc:
        return dict(ok=False, error=str(exc))

B = _load_backend()


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "messages":      [],
    "query_count":   0,
    "resp_ms":       [],
    "docs_loaded":   0,
    "quick_prefix":  "",
    "pinned":        [],
    # Gamification
    "xp":            0,
    "streak":        0,
    "last_date":     "",     # ISO date of last activity
    "mode_counts":   {},     # mode → count
    # Adaptive
    "difficulty_level": "Beginner",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Streak update ─────────────────────────────────────────────────────────────
_today = datetime.date.today().isoformat()
if st.session_state.last_date != _today:
    _yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    if st.session_state.last_date == _yesterday:
        st.session_state.streak += 1
    elif st.session_state.last_date == "":
        st.session_state.streak = 1
    else:
        st.session_state.streak = 1   # reset streak if gap
    st.session_state.last_date = _today


# ══════════════════════════════════════════════════════════════════════════════
#  CRASH GUARD
# ══════════════════════════════════════════════════════════════════════════════
if not B["ok"]:
    st.error(f"**Backend failed to load:**\n\n```\n{B['error']}\n```")
    st.info("Fix: run `pip install -r requirements.txt`, then restart Streamlit.")
    st.stop()

_do_query  = B["query"]
_do_stream = B["stream"]
_ingest    = B["ingest"]
_stats     = B["stats"]
_subjects  = B["subjects"]
_models    = B["models"]
_online    = B["online"]
_detect    = B["detect"]
MODES      = B["modes"]
DEFAULT_MODEL = B["default_model"]

MODE_ICON = {
    "Explanation": "💡", "5-Mark": "✏️", "10-Mark": "📝",
    "Revision": "🔁", "Quiz": "🧪", "Strategy": "🗺️",
}
MODE_XP = {"Explanation": 5, "5-Mark": 7, "10-Mark": 10,
           "Revision": 6, "Quiz": 12, "Strategy": 8}

SPINNERS = [
    "🧠 Thinking…", "📚 Retrieving knowledge…",
    "🔍 Searching your docs…", "⚡ Crafting answer…",
    "🎯 Structuring response…",
]

# ── Helper: adaptive difficulty string ───────────────────────────────────────
def _difficulty_label() -> str:
    n = st.session_state.query_count
    if n < 6:   return "Beginner"
    if n < 16:  return "Intermediate"
    return "Advanced"

def _difficulty_css() -> str:
    d = _difficulty_label()
    return {"Beginner": "lvl-beginner", "Intermediate": "lvl-intermediate",
            "Advanced": "lvl-advanced"}.get(d, "lvl-beginner")

# ── Helper: smart suggestion ──────────────────────────────────────────────────
def _smart_suggestion(mode: str, query_count: int) -> str:
    """Return a contextual learning nudge."""
    mc = st.session_state.mode_counts
    if mode == "Explanation" and mc.get("Explanation", 0) >= 2:
        return "💡 You've read a few explanations. Try **Quiz** mode to test yourself!"
    if mode == "Quiz" and mc.get("Quiz", 0) >= 1:
        return "🔁 Good quiz work! Follow up with **Revision** mode to consolidate."
    if mode == "5-Mark" and mc.get("10-Mark", 0) == 0 and query_count >= 5:
        return "📝 Ready for more depth? Try **10-Mark** mode for longer exam answers."
    if mode == "Revision" and mc.get("Quiz", 0) == 0:
        return "🧪 After revising, test yourself with **Quiz** mode!"
    return ""


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Brand header
    st.markdown("""
    <div style="padding:4px 0 12px">
      <div style="font-size:1.25rem;font-weight:800;letter-spacing:-.04em;
                  background:linear-gradient(90deg,#dde6f0,#5b9cf6);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent">
        🧠 ExamRAG
      </div>
      <div style="font-size:.67rem;color:#3a4f65;margin-top:1px">powered by Viyaan AI</div>
    </div>
    """, unsafe_allow_html=True)

    # Gamification badges row
    xp     = st.session_state.xp
    streak = st.session_state.streak
    dlvl   = _difficulty_label()
    dcss   = _difficulty_css()
    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px">'
        f'<span class="xpbadge">⚡ {xp} XP</span>'
        f'<span class="streakbadge">🔥 {streak}-day streak</span>'
        f'<span class="lvlbadge {dcss}">{dlvl}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Ollama status
    ollama_ok = _online()
    if ollama_ok:
        st.markdown('<span class="spill spill-on"><span class="sdot sdot-on"></span>Ollama · Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="spill spill-off"><span class="sdot sdot-off"></span>Ollama · Offline</span>', unsafe_allow_html=True)
        st.warning("```bash\nollama serve\nollama pull phi\n```")

    st.divider()

    # Model
    avail = _models() if ollama_ok else [DEFAULT_MODEL]
    if not avail:
        avail = [DEFAULT_MODEL]
    sel_model = st.selectbox("🤖 Model", avail, key="sb_model")

    # Mode
    sel_mode = st.selectbox("📝 Exam Mode", MODES, key="sb_mode")

    # Auto-detect toggle
    auto_detect = st.toggle("🔮 Auto-detect mode", value=True, key="sb_auto")

    # Subject
    try:
        subs = _subjects()
    except Exception:
        subs = ["All"]
    sel_subject = st.selectbox("🔖 Subject filter", subs, key="sb_subj")

    # Top-k
    top_k = st.slider("🔍 Retrieval depth (top-k)", 1, 3, 3, key="sb_topk",
                      help="Kept ≤3 for 8 GB RAM performance")

    st.divider()

    # Index stats
    try:
        idx   = _stats()
        vecs  = idx.get("total_vectors", 0)
        slist = ", ".join(s for s in idx.get("subjects", ["–"]) if s != "All") or "–"
    except Exception:
        vecs, slist = 0, "–"

    c1, c2, c3 = st.columns(3)
    c1.metric("Vectors", f"{vecs:,}")
    c2.metric("Queries", st.session_state.query_count)
    avg_ms = int(sum(st.session_state.resp_ms) / len(st.session_state.resp_ms)) if st.session_state.resp_ms else 0
    c3.metric("Avg ms", avg_ms)

    st.divider()

    # Upload
    st.markdown("**📎 Upload Document**")
    uploaded = st.file_uploader(
        "PDF or TXT",
        type=["pdf", "txt"],
        label_visibility="collapsed",
        key="uploader",
    )
    if uploaded and st.button("⚡ Ingest", use_container_width=True):
        with st.spinner(f"Processing {uploaded.name}…"):
            try:
                raw = uploaded.read()
                if not raw:
                    st.error("File appears to be empty.")
                else:
                    n = _ingest(raw, filename=uploaded.name)
                    st.success(f"✅ {n} chunks added from '{uploaded.name}'")
                    st.session_state.docs_loaded += 1
                    st.cache_resource.clear()
                    st.rerun()
            except Exception as exc:
                st.error(f"Ingest failed: {exc}")

    st.divider()

    # Chat actions
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑 Clear", use_container_width=True):
            for k in ("messages", "resp_ms", "quick_prefix"):
                st.session_state[k] = [] if k != "quick_prefix" else ""
            st.session_state.query_count = 0
            st.rerun()
    with col_b:
        if st.session_state.messages:
            st.download_button(
                "⬇ Export",
                data=json.dumps(st.session_state.messages, indent=2, ensure_ascii=False),
                file_name="examrag_chat.json",
                mime="application/json",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_chat, tab_dash, tab_pins = st.tabs(["💬 Chat", "📊 Dashboard", "📌 Pinned"])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1: CHAT
# ─────────────────────────────────────────────────────────────────────────────
with tab_chat:

    # Header
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;
                padding:.4rem 0 .55rem;border-bottom:1px solid #1c2840;
                margin-bottom:.75rem">
      <span style="font-size:1.8rem;background:linear-gradient(135deg,#5b9cf6,#a78bfa,#34d399);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent">🧠</span>
      <div>
        <div style="font-size:1.35rem;font-weight:800;letter-spacing:-.04em;
                    background:linear-gradient(90deg,#dde6f0 55%,#5b9cf6);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent">
          ExamRAG — Powered by Viyaan AI
        </div>
        <div style="font-size:.68rem;color:#3a4f65">
          Train Your Mind. Master Your Exams.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick-action buttons
    qa_map = [
        ("💡 Explain",  "Explain in detail: "),
        ("✏️ 5-Mark",   "Write a 5-mark answer for: "),
        ("📝 10-Mark",  "Write a 10-mark answer for: "),
        ("🔁 Revise",   "Give revision notes for: "),
        ("🧪 Quiz",     "Generate a quiz on: "),
        ("🗺️ Strategy", "Give study strategy for: "),
    ]
    qa_cols = st.columns(6)
    for col, (label, prefix) in zip(qa_cols, qa_map):
        with col:
            if st.button(label, use_container_width=True, key=f"qa_{label}"):
                st.session_state.quick_prefix = prefix
                st.rerun()

    # Empty-state hint
    if not st.session_state.messages:
        if vecs == 0:
            st.info(
                "👈 **Upload a PDF** in the sidebar to index your study material.\n\n"
                "Everything runs **100% locally** — your data never leaves this machine."
            )
        else:
            st.info(f"📚 **{vecs:,} vectors indexed** across: *{slist}*.\n\nAsk anything below!")

    # Render chat history
    for i, msg in enumerate(st.session_state.messages):
        is_user = msg["role"] == "user"
        with st.chat_message("human" if is_user else "assistant",
                             avatar="🧑" if is_user else "🧠"):
            if is_user:
                st.markdown(msg["content"])
            else:
                meta     = msg.get("meta", {})
                eff_mode = meta.get("mode", "")
                fallback = meta.get("fallback", False)
                auto_det = meta.get("auto_detected", False)

                icon_str = MODE_ICON.get(eff_mode, "🤖")
                fb_tag   = '<span class="ftag">⚡ no docs</span>' if fallback else ""
                ad_tag   = '<span class="atag">🔮 auto</span>'   if auto_det else ""
                st.markdown(
                    f'<span class="mtag">{icon_str} {eff_mode}</span>{fb_tag}{ad_tag}',
                    unsafe_allow_html=True,
                )
                st.markdown(msg["content"])

                kps = meta.get("key_points", [])
                if kps:
                    with st.expander("📋 Key Points", expanded=True):
                        for kp in kps:
                            st.markdown(f"▸ {kp}")

                tip = meta.get("exam_tip", "")
                if tip:
                    st.markdown(
                        f'<div class="tipbox">💡 <b>Exam Tip:</b> {tip}</div>',
                        unsafe_allow_html=True,
                    )

                # Smart suggestion
                suggestion = meta.get("suggestion", "")
                if suggestion:
                    st.markdown(
                        f'<div class="suggest">🎯 {suggestion}</div>',
                        unsafe_allow_html=True,
                    )

                # XP earned
                xp_earned = meta.get("xp_earned", 0)
                if xp_earned:
                    st.caption(f"⚡ +{xp_earned} XP earned")

                srcs = meta.get("sources", [])
                if srcs:
                    with st.expander(f"📄 Sources ({len(srcs)})"):
                        for s in srcs:
                            score_pct = int(s.get("score", 0) * 100)
                            st.markdown(
                                f'<span class="schip">📄 {s["source"]} p.{s["page"]} · {score_pct}%</span>',
                                unsafe_allow_html=True,
                            )
                            st.caption(s.get("preview", ""))

                # Pin button
                _, pcol = st.columns([10, 2])
                with pcol:
                    if st.button("📌", key=f"pin_{i}", help="Pin this answer"):
                        entry = msg["content"]
                        if entry not in st.session_state.pinned:
                            st.session_state.pinned.append(entry)
                            st.toast("Pinned!", icon="📌")
                        else:
                            st.toast("Already pinned.", icon="ℹ️")

    # ── Chat input ────────────────────────────────────────────────────────────
    placeholder = (
        st.session_state.quick_prefix + "…"
        if st.session_state.quick_prefix
        else f"Ask anything ({sel_mode} mode active)"
    )
    user_input = st.chat_input(placeholder)

    # ── Process input (fires ONCE per submit) ─────────────────────────────────
    if user_input:
        if st.session_state.quick_prefix:
            final_q = st.session_state.quick_prefix + user_input
            st.session_state.quick_prefix = ""
        else:
            final_q = user_input

        ts = datetime.datetime.now().strftime("%H:%M")

        # Append user message
        st.session_state.messages.append({"role": "user", "content": final_q, "meta": {}})

        # Detect mode
        effective_mode = _detect(final_q) if auto_detect else sel_mode

        # Track mode usage
        mc = st.session_state.mode_counts
        mc[effective_mode] = mc.get(effective_mode, 0) + 1

        t0        = time.time()
        full_text = ""
        meta_payload = {}

        with st.chat_message("assistant", avatar="🧠"):
            icon_str = MODE_ICON.get(effective_mode, "🤖")
            st.markdown(
                f'<span class="mtag">{icon_str} {effective_mode}</span>',
                unsafe_allow_html=True,
            )

            stream_ph  = st.empty()
            spinner_msg = random.choice(SPINNERS)

            try:
                gen   = _do_stream(
                    final_q,
                    mode=effective_mode if not auto_detect else None,
                    top_k=top_k,
                    subject_filter=sel_subject,
                    model=sel_model,
                    query_count=st.session_state.query_count,
                )

                # First event must be "meta"
                first = next(gen, None)
                if first and first["type"] == "meta":
                    meta_payload = first["data"]

                # Stream tokens
                with st.spinner(spinner_msg):
                    for event in gen:
                        if event["type"] == "token":
                            full_text += event["data"]
                            stream_ph.markdown(full_text + "▌")
                        elif event["type"] == "done":
                            break

                stream_ph.markdown(full_text)

            except Exception as exc:
                full_text = f"⚠️ **Error:** {exc}\n\nMake sure Ollama is running: `ollama serve`"
                stream_ph.markdown(full_text)
                meta_payload = {"fallback": True, "sources": [], "mode": effective_mode}

        elapsed_ms = int((time.time() - t0) * 1000)
        st.session_state.resp_ms.append(elapsed_ms)
        st.session_state.query_count += 1

        # Parse structured fields from streamed text
        kp_match = _re.search(r"KEYPOINTS:\s*(.+?)(?:\nEXAMTIP:|$)", full_text, _re.DOTALL)
        kps: list = []
        if kp_match:
            raw_kp = kp_match.group(1).strip()
            if "|" in raw_kp and "\n" not in raw_kp.strip():
                candidates = [p.strip() for p in raw_kp.split("|")]
            else:
                candidates = raw_kp.split("\n")
            for line in candidates:
                line = _re.sub(r"^[-•*\d.)\s]+", "", line).strip()
                if len(line) > 8:
                    kps.append(line)

        tip_match   = _re.search(r"EXAMTIP:\s*(.+?)$", full_text, _re.DOTALL)
        exam_tip    = tip_match.group(1).strip()[:250] if tip_match else ""
        display_ans = _re.split(r"\nKEYPOINTS:|\nEXAMTIP:", full_text)[0].strip()

        # Gamification: award XP
        xp_earned = MODE_XP.get(effective_mode, 5)
        st.session_state.xp += xp_earned

        # Smart suggestion
        suggestion = _smart_suggestion(effective_mode, st.session_state.query_count)

        # Append assistant message
        st.session_state.messages.append({
            "role":    "assistant",
            "content": display_ans,
            "meta": {
                "mode":          meta_payload.get("mode", effective_mode),
                "fallback":      meta_payload.get("fallback", False),
                "auto_detected": auto_detect,
                "sources":       meta_payload.get("sources", []),
                "context":       meta_payload.get("context", ""),
                "key_points":    kps,
                "exam_tip":      exam_tip,
                "suggestion":    suggestion,
                "xp_earned":     xp_earned,
                "ts":            ts,
                "elapsed_ms":    elapsed_ms,
            },
        })

        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab_dash:
    st.markdown("### 📊 Session Dashboard")

    # Performance metrics
    m1, m2, m3, m4 = st.columns(4)
    total_q  = st.session_state.query_count
    avg_ms2  = int(sum(st.session_state.resp_ms) / len(st.session_state.resp_ms)) if st.session_state.resp_ms else 0
    fastest  = min(st.session_state.resp_ms) if st.session_state.resp_ms else 0
    slowest  = max(st.session_state.resp_ms) if st.session_state.resp_ms else 0
    m1.metric("Total Queries",  total_q)
    m2.metric("Avg Response",   f"{avg_ms2} ms")
    m3.metric("Fastest",        f"{fastest} ms")
    m4.metric("Slowest",        f"{slowest} ms")

    st.divider()

    # Gamification panel
    st.markdown("**🎮 Your Progress**")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("⚡ XP",       st.session_state.xp)
    g2.metric("🔥 Streak",   f"{st.session_state.streak} days")
    g3.metric("📚 Level",    _difficulty_label())
    g4.metric("📌 Pinned",   len(st.session_state.pinned))

    # XP progress to next level
    n = st.session_state.query_count
    if n < 6:
        next_n, lvl_label = 6,  "Intermediate"
        pct = n / 6
    elif n < 16:
        next_n, lvl_label = 16, "Advanced"
        pct = (n - 6) / 10
    else:
        next_n, lvl_label = n,  "Max Level 🏆"
        pct = 1.0
    st.caption(f"Progress to {lvl_label}: {n}/{next_n} queries")
    st.progress(min(pct, 1.0))

    st.divider()

    # Mode usage breakdown
    mode_counts = st.session_state.mode_counts
    if mode_counts:
        st.markdown("**Mode Usage**")
        total_u = sum(mode_counts.values()) or 1
        for mname, cnt in sorted(mode_counts.items(), key=lambda x: -x[1]):
            pct_u = cnt / total_u
            st.markdown(f"`{MODE_ICON.get(mname,'📝')} {mname}` — **{cnt}** queries")
            st.progress(pct_u)
    else:
        st.info("No queries yet. Ask something in the Chat tab!")

    st.divider()

    # Knowledge index
    st.markdown("**Knowledge Index**")
    try:
        idx_data  = _stats()
        total_v   = idx_data.get("total_vectors", 0)
        subj_list = [s for s in idx_data.get("subjects", []) if s != "All"]
    except Exception:
        total_v, subj_list = 0, []

    vi1, vi2 = st.columns(2)
    vi1.metric("Vectors",  f"{total_v:,}")
    vi2.metric("Subjects", len(subj_list))
    if subj_list:
        st.markdown("Indexed subjects: " + " · ".join(f"`{s}`" for s in subj_list))

    st.divider()

    # System
    st.markdown("**System**")
    si1, si2, si3 = st.columns(3)
    si1.metric("Model",    sel_model.split(":")[0] if ":" in sel_model else sel_model)
    si2.metric("Top-k",    top_k)
    si3.metric("Docs in",  st.session_state.docs_loaded)

    # Response time chart
    if len(st.session_state.resp_ms) > 1:
        st.divider()
        st.markdown("**Response Times (ms)**")
        import pandas as pd
        df = pd.DataFrame({
            "Query": list(range(1, len(st.session_state.resp_ms) + 1)),
            "ms":    st.session_state.resp_ms,
        }).set_index("Query")
        st.line_chart(df, color="#5b9cf6")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3: PINNED ANSWERS
# ─────────────────────────────────────────────────────────────────────────────
with tab_pins:
    st.markdown("### 📌 Pinned Answers")

    if not st.session_state.pinned:
        st.info("No pinned answers yet. Click **📌** on any response in the Chat tab to save it here.")
    else:
        st.caption(f"{len(st.session_state.pinned)} answer(s) pinned")
        for pi, pinned_text in enumerate(st.session_state.pinned):
            label = pinned_text[:70].replace("\n", " ") + ("…" if len(pinned_text) > 70 else "")
            with st.expander(f"📌 {pi + 1}. {label}", expanded=False):
                st.markdown(pinned_text)
                _, col_del = st.columns([8, 2])
                with col_del:
                    if st.button("✕ Remove", key=f"unpin_{pi}"):
                        st.session_state.pinned.pop(pi)
                        st.rerun()
        if st.button("🗑 Clear All Pins"):
            st.session_state.pinned = []
            st.rerun()
