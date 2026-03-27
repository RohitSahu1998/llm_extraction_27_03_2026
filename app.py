import streamlit as st
import os
import tempfile
import pandas as pd
import json

from ocr_engine import PaddleOCREngine
from qwen_engine import QwenExtractor
from matcher import highlight_and_save_pdf

# UI Configuration
st.set_page_config(page_title="Document AI Extractor", layout="wide", page_icon="📄")

st.title("📄 Intelligent Document Extraction Pipeline")
st.markdown("Upload a Document (PDF/Image) to instantly extract structured semantic fields, match them precisely to OCR coordinates, and generate a highlighted verification PDF.")

# Cache the AI models so they don't reload every time the user clicks a button!
@st.cache_resource(show_spinner=False)
def load_ai_models():
    with st.spinner("Loading Vision-Language Model and OCR Engines (First run only)..."):
        qwen = QwenExtractor()
        
        # NOTE: If your local machine doesn't have an Nvidia GPU installed, you may need to set use_gpu=False
        try:
            ocr = PaddleOCREngine(use_gpu=True) 
        except Exception:
            ocr = PaddleOCREngine(use_gpu=False)
            
        return qwen, ocr

uploaded_file = st.file_uploader("Upload an Invoice, Claim, or Form", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    
    # Save the uploaded file temporarily so the backend engines can read it from a path
    file_bytes = uploaded_file.read()
    file_extension = os.path.splitext(uploaded_file.name)[1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
        
    st.success(f"**{uploaded_file.name}** uploaded safely to memory!")
    
    if st.button("🚀 Run AI Extraction Pipeline", use_container_width=True, type="primary"):
        try:
            # 1. Load Models
            qwen_extractor, ocr_engine = load_ai_models()
            
            # 2. Extract Logic
            st.markdown("### Pipeline Execution Steps:")
            
            with st.spinner("🧠 Step 1/3: Running Qwen 2.5 Vision-Language Model..."):
                qwen_data = qwen_extractor.extract_data(temp_path)
                st.success("✅ Step 1: Semantic understanding complete!")
                
            with st.spinner("🔍 Step 2/3: Running PaddleOCR engine across all pages..."):
                ocr_data = ocr_engine.extract_text_with_confidence(temp_path)
                st.success("✅ Step 2: Pixel-level word extraction complete!")
                
            with st.spinner("🔗 Step 3/3: Running Anchor & Spatial Matching to link Qwen with OCR..."):
                output_pdf = temp_path + "_highlighted.pdf"
                output_csv = output_pdf.replace(".pdf", ".csv").replace(".jpg", ".csv")
                
                # Run the matcher which draws the boxes, saves the PDF, and outputs the CSV
                highlight_and_save_pdf(temp_path, qwen_data, ocr_data, output_pdf)
                st.success("✅ Step 3: Visual highlighting and alignment CSV generated!")
                
            # --- Display Results ---
            st.divider()
            st.header("Results Dashboard")
            
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                st.subheader("Raw LLM Extraction (JSON)")
                st.json(qwen_data)
                
            with col2:
                st.subheader("Final Matched Entities")
                if os.path.exists(output_csv):
                    df = pd.read_csv(output_csv)
                    # Color formatting trick for highlighting missed values in red
                    def highlight_missed(row):
                        if row["OCR_Matched_Text"] == "NO MATCH":
                            return ['background-color: #ffcccc'] * len(row)
                        return [''] * len(row)
                        
                    st.dataframe(df.style.apply(highlight_missed, axis=1), use_container_width=True, height=500)
                else:
                    st.warning("CSV Data Not Found.")
                    
            # --- Download Buttons ---
            st.divider()
            st.subheader("📥 Download Generated Artifacts")
            d_col1, d_col2 = st.columns(2)
            
            with d_col1:
                if os.path.exists(output_pdf):
                    with open(output_pdf, "rb") as f:
                        # Ensures the downloaded file format makes sense (pdf if available, otherwise original format)
                        dl_name = f"Verified_{uploaded_file.name}"
                        if not dl_name.lower().endswith('.pdf'):
                            dl_name = dl_name.rsplit('.', 1)[0] + '.pdf'
                            
                        st.download_button(
                            label="Download Highlighted Document",
                            data=f,
                            file_name=dl_name,
                            mime="application/pdf" if dl_name.endswith('.pdf') else "image/jpeg",
                            type="primary"
                        )
                        
            with d_col2:
                if os.path.exists(output_csv):
                    with open(output_csv, "rb") as f:
                        st.download_button(
                            label="Download Data Table (CSV)",
                            data=f,
                            file_name=f"Data_{uploaded_file.name}.csv",
                            mime="text/csv",
                            type="primary"
                        )
                        
        except Exception as e:
            st.error(f"Pipeline crashed: {str(e)}")
            
        finally:
            # We don't delete temp_path immediately so downloads can fetch it, 
            # Streamlit clears its temp folder automatically on session resets anyway!
            pass
