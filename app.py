"""
Wasel v4 Pro: Intelligent Sign Language Translator
Clean, modular Streamlit interface (< 200 lines).
Replaces 1270-line monolithic app.py from v3.
"""

import os
import tempfile
import logging
import streamlit as st

# ─── Page Config (must be first) ───
st.set_page_config(page_title="Wasel v4 Pro", page_icon="🤟", layout="wide")

# ─── Logging ───
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Environment ───
DATA_DIR = os.environ.get("WASEL_DATA_DIR", os.path.join(tempfile.gettempdir(), "wasel_v4_data"))
os.makedirs(DATA_DIR, exist_ok=True)

# ═══════════════════════════════════════════
# ─── CACHED RESOURCE LOADERS ───
# ═══════════════════════════════════════════

@st.cache_resource
def get_engine():
    from backend.engine import WaselEngine
    legacy_pkl = os.path.join(os.getcwd(), "assets", "psl_classifier.pkl")
    engine = WaselEngine(
        data_dir=DATA_DIR,
        legacy_model_path=legacy_pkl if os.path.exists(legacy_pkl) else None
    )
    return engine

@st.cache_resource
def get_vocabulary():
    from backend.vocabulary import VocabularyManager
    return VocabularyManager(include_extended=True)

@st.cache_resource
def get_renderer():
    from backend.digital_human import DigitalHumanRenderer
    return DigitalHumanRenderer()

def auto_setup(engine, vocab):
    """Auto-build on first run, instant on subsequent runs."""
    if engine.classifier and len(engine.landmark_dict) > 0:
        return True

    with st.status("🚀 **First-Time Setup** — Building AI Engine...", expanded=True) as status:
        if len(engine.landmark_dict) == 0:
            st.write("📥 **Step 1/3:** Downloading PSL reference videos...")
            try:
                import sign_language_translator as slt
                translator = slt.models.ConcatenativeSynthesis(
                    text_language="urdu", sign_language="psl", sign_format="vid"
                )
                engine.build_vocabulary(translator=translator, word_map=vocab.get_core())
            except Exception as e:
                st.write(f"⚠️ SLT Library unavailable ({e}). Using cached data if available.")
            st.write(f"✅ DNA extracted for **{len(engine.landmark_dict)}** words.")
        else:
            st.write(f"✅ **Step 1/3:** {len(engine.landmark_dict)} words loaded from cache.")

        if not engine.classifier:
            st.write("🧠 **Step 2/3:** Training classifier...")
            engine.train()
            st.write("✅ Classifier trained and saved.")
        else:
            st.write("✅ **Step 2/3:** Classifier loaded from cache.")

        words = engine.get_available_words()
        st.write(f"🔍 **Step 3/3:** {len(words)} words ready: {', '.join(words)}")
        status.update(label="✅ **Setup Complete!**", state="complete")
    return True


# ═══════════════════════════════════════════
# ─── MAIN APP ───
# ═══════════════════════════════════════════

def main():
    # ─── CSS ───
    st.markdown("""<style>
        .main { background-color: #0f172a; }
        h1, h2, h3 { color: #38bdf8 !important; }
        .stButton>button { border-radius: 10px; border: 1px solid #4f46e5; }
    </style>""", unsafe_allow_html=True)

    st.title("🤟 Wasel v4 Pro")
    st.markdown("**Next-Gen Pakistan Sign Language Translator** — YOLO + TensorFlow")
    st.divider()

    # Load engines
    with st.spinner("⏳ Loading AI engines..."):
        engine = get_engine()
        vocab = get_vocabulary()
        renderer = get_renderer()

    if not auto_setup(engine, vocab):
        st.error("❌ Setup failed.")
        st.stop()

    # ─── Sidebar ───
    st.sidebar.success(f"💎 **Wasel v4 Pro** | Elite Studio")
    st.sidebar.success(f"🧠 **Engine:** {engine.backend['pose']} + {engine.backend['classifier']}")
    st.sidebar.info(f"📚 **Vocabulary:** {len(engine.get_available_words())} words")

    # ─── Tabs ───
    tab_text, tab_video = st.tabs(["📝 Text → Video", "🎥 Video → Text"])

    # ─── TEXT → VIDEO ───
    with tab_text:
        text_input = st.text_input("Enter English text:", placeholder="good apple pakistan")
        if st.button("🔄 Translate to PSL", key="translate_btn"):
            if text_input:
                words = text_input.lower().split()
                dna_list = []

                for w in words:
                    dna = engine.get_word_dna(w)
                    if dna is not None:
                        dna_list.append(dna)
                        st.success(f"✅ {w}")
                    else:
                        st.warning(f"⚠️ '{w}' not in vocabulary. Skipping.")

                if dna_list:
                    out_path = os.path.join(tempfile.gettempdir(), "wasel_v4_output.mp4")
                    renderer.stitch_and_render(dna_list, out_path)
                    st.video(out_path)

    # ─── VIDEO → TEXT ───
    with tab_video:
        col_upload, col_live = st.columns(2)

        with col_upload:
            st.subheader("📁 Upload Video")
            uploaded = st.file_uploader("Upload sign clip:", type=["mp4", "avi", "mov"])
            if uploaded:
                tmp = os.path.join(tempfile.gettempdir(), "wasel_v4_input.mp4")
                with open(tmp, "wb") as f:
                    f.write(uploaded.read())

                with st.spinner("🧠 Analyzing..."):
                    labels, conf = engine.predict_sentence(tmp)

                if labels:
                    st.success(f"🎯 **Recognized:** {' '.join(labels)}")
                    st.metric("Confidence", f"{conf:.1f}%")
                else:
                    st.warning("⚠️ No signs detected. Try a clearer video.")

        with col_live:
            st.subheader("🌐 Live Stream")
            try:
                from streamlit_webrtc import webrtc_streamer
                from streaming.webrtc_hub import SignStreamProcessor

                webrtc_streamer(
                    key="wasel_v4_live",
                    video_processor_factory=lambda: SignStreamProcessor(engine=engine),
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={"video": True, "audio": False},
                )
            except ImportError:
                st.info("📦 `streamlit-webrtc` required for live streaming.")
            except Exception as e:
                st.error(f"❌ Stream error: {e}")

    # ─── Footer ───
    st.divider()
    st.caption("Designed by Ahmed Eltaweel | AI Architect @ Konecta 🚀")

if __name__ == "__main__":
    main()
