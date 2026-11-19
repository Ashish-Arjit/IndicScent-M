"""
member5/app.py
Interactive Streamlit Application for IndicSent-M: Multimodal Sentiment Analysis.
Provides real-time multimodal inference, confidence scores, and modality conflict detection.
"""

import sys
from pathlib import Path
import json
import streamlit as st
import pandas as pd
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR, SAMPLE_MEMES_DIR, APP_CONFIG, CLASS_NAMES
from member5.integration import IndicSentPipeline

# Streamlit Page Configuration
st.set_page_config(
    page_title=APP_CONFIG["title"],
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 12px;
    }
    .badge-positive {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
    }
    .badge-neutral {
        background-color: #f1f5f9;
        color: #475569;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
    }
    .badge-negative {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Cache pipeline instance for fast interactive inference."""
    return IndicSentPipeline()


def render_sentiment_badge(sentiment: str) -> str:
    """Format HTML sentiment pill."""
    sent = str(sentiment).lower()
    if "pos" in sent:
        return f'<span class="badge-positive">POSITIVE</span>'
    elif "neg" in sent:
        return f'<span class="badge-negative">NEGATIVE</span>'
    return f'<span class="badge-neutral">NEUTRAL</span>'


def main():
    pipeline = load_pipeline()
    status = pipeline.get_system_status()

    # Sidebar
    st.sidebar.title("🇮🇳 IndicSent-M")
    st.sidebar.caption("Multimodal Sentiment Analysis for Indian Languages")
    st.sidebar.markdown("---")

    st.sidebar.subheader("System Status")
    for mod, is_loaded in status.items():
        name = mod.replace("_loaded", "").replace("_", " ").title()
        if is_loaded:
            st.sidebar.success(f"✔️ {name} Active")
        else:
            st.sidebar.warning(f"⚠️ {name} Not Trained")

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Bennett University CSE Project**\n\n"
        "• Surya Pratap Singh (Member 1)\n"
        "• B.T. Ashish (Member 2)\n"
        "• CH. Akhil (Member 3)\n"
        "• B. Rushindra Choudary (Member 4)\n"
        "• B. Sai Krishna (Member 5)"
    )

    # Main Navigation Tabs
    tab_demo, tab_batch, tab_eval, tab_conflicts, tab_about = st.tabs([
        "🚀 Live Sentiment Demo",
        "📁 Batch CSV Processing",
        "📊 Model Comparison & Metrics",
        "🔍 Conflict Analysis Explorer",
        "📖 Project Overview"
    ])

    # ==================== TAB 1: LIVE DEMO ====================
    with tab_demo:
        st.header("Multimodal Sentiment Analyzer")
        st.write("Analyze text, meme images, or combined social media posts across English, Hindi, and Hinglish.")

        col_input_left, col_input_right = st.columns([1.2, 1])

        with col_input_left:
            st.subheader("1. Text Input")
            preset = st.selectbox(
                "Choose an example caption or enter custom text:",
                [
                    "Custom Input",
                    "Passed semester exam without backlogs! Party toh banti hai bhai! (Positive)",
                    "Today temperature in campus is 32 degrees celsius. (Neutral)",
                    "WiFi disconnected right in the middle of online coding test! (Negative)",
                    "Great job guys, broken code pushed straight to master on Friday! (Sarcastic/Conflict)"
                ]
            )

            if preset == "Custom Input":
                text_input = st.text_area(
                    "Enter caption / meme text (English, Hindi, Hinglish):",
                    placeholder="E.g., Bhai ka placement ho gaya Google mein! Bohot khush hain!"
                )
            else:
                text_input = preset.rsplit(" (", 1)[0]
                st.text_area("Caption:", value=text_input, disabled=True)

        with col_input_right:
            st.subheader("2. Image / Meme Input")
            uploaded_file = st.file_uploader("Upload meme image (JPG/PNG)", type=["jpg", "jpeg", "png"])
            
            sample_img_choice = None
            sample_files = list(SAMPLE_MEMES_DIR.glob("*.jpg"))
            if sample_files and not uploaded_file:
                sample_options = ["None"] + [f.name for f in sample_files[:6]]
                sample_img_choice = st.selectbox("Or choose a sample meme from dataset:", sample_options)

            image_to_analyze = None
            if uploaded_file is not None:
                image_to_analyze = Image.open(uploaded_file)
                st.image(image_to_analyze, caption="Uploaded Image", use_container_width=True)
            elif sample_img_choice and sample_img_choice != "None":
                chosen_path = SAMPLE_MEMES_DIR / sample_img_choice
                image_to_analyze = Image.open(chosen_path)
                st.image(image_to_analyze, caption=f"Sample Meme: {sample_img_choice}", use_container_width=True)

        st.markdown("---")
        if st.button("🔍 Run Sentiment Analysis", type="primary", use_container_width=True):
            if not text_input and not image_to_analyze:
                st.warning("Please provide at least a text caption or an image.")
            else:
                with st.spinner("Processing modalities and computing representations..."):
                    res = pipeline.analyze(text=text_input, image=image_to_analyze)

                # Results Display
                st.subheader("Prediction Results")
                cols = st.columns(3)

                # Text Prediction Card
                with cols[0]:
                    st.markdown("### 📝 Text-Only")
                    t_pred = res.get("text_prediction")
                    if t_pred and "sentiment" in t_pred:
                        st.markdown(render_sentiment_badge(t_pred["sentiment"]), unsafe_allow_html=True)
                        st.metric("Confidence", f"{t_pred['confidence']*100:.1f}%")
                        st.write("Probabilities:")
                        st.json(t_pred["probabilities"])
                    else:
                        st.info(t_pred.get("status", "No text provided") if t_pred else "No text provided")

                # Image Prediction Card
                with cols[1]:
                    st.markdown("### 🖼️ Image-Only")
                    i_pred = res.get("image_prediction")
                    if i_pred and "sentiment" in i_pred:
                        st.markdown(render_sentiment_badge(i_pred["sentiment"]), unsafe_allow_html=True)
                        st.metric("Confidence", f"{i_pred['confidence']*100:.1f}%")
                        st.write("Probabilities:")
                        st.json(i_pred["probabilities"])
                    else:
                        st.info(i_pred.get("status", "No image provided") if i_pred else "No image provided")

                # Multimodal Prediction Card
                with cols[2]:
                    st.markdown("### ⚡ Multimodal Fusion")
                    m_pred = res.get("multimodal_prediction")
                    if m_pred and "sentiment" in m_pred:
                        st.markdown(render_sentiment_badge(m_pred["sentiment"]), unsafe_allow_html=True)
                        st.metric("Confidence", f"{m_pred['confidence']*100:.1f}%")
                        st.write("Probabilities:")
                        st.json(m_pred["probabilities"])
                    else:
                        st.info(m_pred.get("status", "Provide both text & image") if m_pred else "Provide both text & image")

                # Conflict Detection Alert
                conflict = res.get("conflict_diagnosis")
                if conflict:
                    st.markdown("---")
                    st.subheader("Modality Conflict & Sarcasm Diagnosis")
                    if conflict["is_conflict"]:
                        if conflict["conflict_severity"] == "HIGH":
                            st.error(f"🚨 **High Disagreement Detected!**\n\n{conflict['explanation']}")
                        else:
                            st.warning(f"⚠️ **Moderate Disagreement Detected**\n\n{conflict['explanation']}")
                        
                        if conflict.get("multimodal_resolution"):
                            st.info(f"**Resolution:** {conflict['multimodal_resolution']}")
                    else:
                        st.success(f"✅ **Consensus Reached:** Both modalities agree on '{conflict['text_sentiment']}'.")

    # ==================== TAB 2: BATCH CSV PROCESSING (Task 5.1) ====================
    with tab_batch:
        st.header("Batch Evaluation & Dataset Inference")
        st.write("Upload a CSV file containing social media posts, captions, or memes to evaluate across all models.")

        batch_source = st.radio(
            "Select dataset source:",
            ["Upload Custom CSV", "Load Benchmark Test Split (data/processed/test.csv)"],
            horizontal=True
        )

        df_batch = None
        if batch_source == "Upload Custom CSV":
            uploaded_csv = st.file_uploader("Upload CSV file (must include 'text' column, optional 'image_path'):", type=["csv"])
            if uploaded_csv is not None:
                try:
                    df_batch = pd.read_csv(uploaded_csv)
                    st.success(f"Loaded {len(df_batch)} records from uploaded file.")
                except Exception as e:
                    st.error(f"Error parsing uploaded CSV: {e}")
        else:
            test_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "test.csv"
            if test_path.exists():
                df_batch = pd.read_csv(test_path)
                st.info(f"Loaded {len(df_batch)} test benchmark samples from {test_path.name}")
            else:
                st.warning("Benchmark test.csv not found. Run python main.py --prepare-data first.")

        if df_batch is not None and not df_batch.empty:
            st.subheader("Data Preview")
            st.dataframe(df_batch.head(5), use_container_width=True)

            # Detect text and image columns
            candidate_text_cols = [c for c in df_batch.columns if c.lower() in ["text", "caption", "sentence", "content"]]
            text_col = candidate_text_cols[0] if candidate_text_cols else df_batch.columns[0]
            
            candidate_img_cols = [c for c in df_batch.columns if c.lower() in ["image_path", "image", "img_path", "img"]]
            img_col = candidate_img_cols[0] if candidate_img_cols else None

            c1, c2 = st.columns(2)
            with c1:
                selected_text_col = st.selectbox("Text Column:", df_batch.columns, index=df_batch.columns.get_loc(text_col))
            with c2:
                img_options = ["None"] + list(df_batch.columns)
                default_idx = (img_options.index(img_col)) if img_col in img_options else 0
                selected_img_col = st.selectbox("Image Path Column (optional):", img_options, index=default_idx)

            if st.button("⚡ Process Batch Predictions", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_records = []

                total_rows = len(df_batch)
                for i, row in df_batch.iterrows():
                    status_text.text(f"Processing row {i+1} of {total_rows}...")
                    row_text = str(row[selected_text_col]) if pd.notna(row[selected_text_col]) else None
                    row_img = None
                    if selected_img_col != "None" and pd.notna(row[selected_img_col]):
                        p = Path(str(row[selected_img_col]))
                        if p.exists():
                            row_img = p

                    res = pipeline.analyze(text=row_text, image=row_img)
                    t_res = res.get("text_prediction") or {}
                    i_res = res.get("image_prediction") or {}
                    m_res = res.get("multimodal_prediction") or {}
                    diag = res.get("conflict_diagnosis") or {}

                    row_record = {
                        "row_id": i + 1,
                        "text": row_text,
                        "image_path": str(row_img) if row_img else "N/A",
                        "text_sentiment": t_res.get("sentiment", "N/A"),
                        "text_confidence": t_res.get("confidence", "N/A"),
                        "image_sentiment": i_res.get("sentiment", "N/A"),
                        "image_confidence": i_res.get("confidence", "N/A"),
                        "multimodal_sentiment": m_res.get("sentiment", "N/A"),
                        "multimodal_confidence": m_res.get("confidence", "N/A"),
                        "is_conflict": diag.get("is_conflict", False),
                        "conflict_severity": diag.get("conflict_severity", "NONE"),
                        "multimodal_resolution": diag.get("multimodal_resolution", "N/A")
                    }
                    results_records.append(row_record)
                    progress_bar.progress((i + 1) / total_rows)

                status_text.success(f"Successfully processed {total_rows} records!")
                df_out = pd.DataFrame(results_records)

                # Batch summary metrics
                s1, s2, s3 = st.columns(3)
                s1.metric("Processed Samples", total_rows)
                conflicts_n = int(df_out["is_conflict"].sum())
                s2.metric("Disagreements Detected", f"{conflicts_n} ({conflicts_n/total_rows*100:.1f}%)")
                high_n = int((df_out["conflict_severity"] == "HIGH").sum())
                s3.metric("High Severity Conflicts", high_n)

                st.subheader("Annotated Predictions Table")
                st.dataframe(df_out, use_container_width=True)

                # Download CSV
                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Annotated Predictions CSV",
                    data=csv_bytes,
                    file_name="indicsent_batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # ==================== TAB 3: BENCHMARK & METRICS ====================
    with tab_eval:
        st.header("Comparative Model Evaluation")
        st.write("Genuine evaluation metrics computed on the stratified test split.")

        comp_csv = RESULTS_DIR / "model_comparison.csv"
        if comp_csv.exists():
            df_comp = pd.read_csv(comp_csv)
            st.dataframe(df_comp, use_container_width=True)

            # Chart
            if "Accuracy (%)" in df_comp.columns and df_comp["Accuracy (%)"].notna().any():
                st.subheader("Accuracy Comparison")
                st.bar_chart(df_comp.set_index("Model Approach")["Accuracy (%)"])
        else:
            st.info("Benchmark comparison not generated yet. Run: `python main.py --evaluate-all`")

        # Display Confusion Matrices if available
        st.subheader("Confusion Matrices")
        cm_cols = st.columns(3)
        for idx, (m_name, fname) in enumerate([
            ("Text-Only", "text_metrics.json"),
            ("Image-Only", "image_metrics.json"),
            ("Multimodal", "multimodal_metrics.json")
        ]):
            fpath = RESULTS_DIR / fname
            with cm_cols[idx]:
                st.markdown(f"**{m_name}**")
                if fpath.exists():
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    cm = data.get("confusion_matrix")
                    if cm:
                        df_cm = pd.DataFrame(cm, index=["True Neg", "True Neu", "True Pos"], columns=["Pred Neg", "Pred Neu", "Pred Pos"])
                        st.table(df_cm)
                        st.caption(f"Macro-F1: {data.get('macro_f1', 0):.4f}")
                else:
                    st.caption("No metrics found.")

    # ==================== TAB 4: CONFLICT EXPLORER ====================
    with tab_conflicts:
        st.header("Test-Set Conflict & Disagreement Analysis")
        st.write("Browse test samples where text-only and image-only models produced divergent predictions.")

        conf_csv = RESULTS_DIR / "conflict_analysis.csv"
        conf_sum = RESULTS_DIR / "conflict_summary.json"

        if conf_sum.exists():
            with open(conf_sum, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            k1, k2, k3 = st.columns(3)
            k1.metric("Total Test Samples", s_data.get("total_test_samples", 0))
            k2.metric("Disagreements Flagged", f"{s_data.get('disagreeing_samples', 0)} ({s_data.get('conflict_rate_percent', 0)}%)")
            k3.metric("High Severity Clashes", s_data.get("high_severity_conflicts", 0))

        if conf_csv.exists():
            df_conf = pd.read_csv(conf_csv)
            only_clashes = st.checkbox("Show only conflicting samples", value=True)
            if only_clashes:
                df_conf = df_conf[df_conf["is_conflict"] == True]
            st.dataframe(df_conf, use_container_width=True)
        else:
            st.info("Conflict analysis not run yet. Run: `python member4/conflict_detection.py`")

    # ==================== TAB 5: PROJECT OVERVIEW ====================
    with tab_about:
        st.header("Project Architecture: IndicSent-M")
        st.markdown("""
        **IndicSent-M** investigates multimodal sentiment classification for Indian social media content.
        
        ### The Three Approaches Compared
        1. **Text-Only:** N-gram TF-IDF vectorization with class-balanced Logistic Regression / IndicBERT embeddings.
        2. **Image-Only:** Pretrained ResNet-18 visual feature extraction (512 dimensions) with calibrated classification head.
        3. **Multimodal Fusion:** Early concatenation of visual and textual representations with StandardScaling and unified classification.

        ### Team Roles & Responsibilities
        - **Member 1 (Surya Pratap Singh):** Dataset ingestion, cleaning, label mapping, stratified splits, and data health reporting.
        - **Member 2 (B.T. Ashish):** Code-mixed text preprocessing, emoji preservation, TF-IDF feature extraction, and text classifier.
        - **Member 3 (CH. Akhil):** Image tensor transformation, ResNet-18 visual embedding extraction, and image classifier.
        - **Member 4 (B. Rushindra Choudary):** Early multimodal feature fusion, conflict detection pipeline, and 3-way comparative evaluation.
        - **Member 5 (B. Sai Krishna):** Streamlit interactive UI, unified inference pipeline, integration testing, and documentation.
        """)


if __name__ == "__main__":
    main()
