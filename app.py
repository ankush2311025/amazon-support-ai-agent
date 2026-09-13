import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from support_agent import run_agent


st.set_page_config(
    page_title="Amazon Support AI Agent",
    page_icon="📦",
    layout="wide"
)


# ---------- Styling ----------

st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }

    .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 1.5rem;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        min-height: 110px;
    }

    .reply-box {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.05);
        line-height: 1.6;
    }

    .evidence-box {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.20);
        margin-bottom: 0.8rem;
    }

    .small-text {
        color: #777;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------- Header ----------

st.markdown("""
<div class="hero">
    <h1>📦 Amazon Support AI Agent</h1>
    <p>
        AI-assisted customer support triage, reply drafting,
        historical evidence retrieval, and escalation decisions.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------- Input ----------

st.subheader("Customer message")

message = st.text_area(
    "Enter a customer support message",
    placeholder="Example: My package says delivered but I haven't received it.",
    height=130,
    label_visibility="collapsed"
)

analyze = st.button(
    "Analyze & Draft Reply",
    type="primary",
    use_container_width=True
)


# ---------- Run Agent ----------

if analyze:
    if not message.strip():
        st.warning("Please enter a customer message.")
        st.stop()

    with st.spinner("Analyzing message and retrieving similar historical cases..."):
        result = run_agent(message.strip())

    st.divider()

    # ---------- Summary ----------

    st.subheader("Agent decision")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.caption("INTENT")
        st.markdown(f"### {result['intent'].replace('_', ' ').title()}")
        st.caption(f"Method: {result['classification_method']}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.caption("CONFIDENCE")
        st.markdown(f"### {result['classification_confidence']:.0%}")
        st.caption("Classification confidence")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.caption("ACTION")
        if result["decision"] == "escalate":
            st.markdown("### 🔴 Escalate")
        else:
            st.markdown("### 🟢 Auto-handle")
        st.caption(result["escalation_reason"])
        st.markdown("</div>", unsafe_allow_html=True)


    # ---------- Reply ----------

    st.subheader("Draft reply")

    st.markdown(
        f'<div class="reply-box">{result["reply"]}</div>',
        unsafe_allow_html=True
    )


    # ---------- Evidence ----------

    st.subheader("Historical evidence")

    if not result["evidence"]:
        st.info("No historical evidence found.")
    else:
        for i, item in enumerate(result["evidence"], start=1):
            with st.expander(
                f"Case {i} · similarity {item['similarity']:.3f}"
            ):
                st.markdown("**Historical customer message**")
                st.write(item["customer_message"])

                st.markdown("**Historical Amazon response**")
                st.write(item["historical_response"])

                st.caption(f"Case ID: {item['case_id']}")


# ---------- Footer ----------

st.divider()

st.caption(
    "Prototype built for the Hiver SDE Intern take-home assignment. "
    "Historical responses are used as grounding evidence; the agent does "
    "not blindly copy them."
)
