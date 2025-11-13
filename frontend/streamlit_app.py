"""
Streamlit frontend for RAG Question Generator
"""
import streamlit as st
import requests
from typing import List, Dict
import json

# API endpoint
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Question Generator",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG-Based IIT JEE Question Generator")
st.markdown("Generate unique questions from your PDF question papers using RAG")

# Sidebar
with st.sidebar:
    st.header("Settings")
    question_count = st.slider(
        "Number of Questions",
        min_value=1,
        max_value=50,
        value=10,
        step=1
    )
    
    if st.button("🔄 Check API Status"):
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.success(f"✅ API is running")
                st.info(f"Total questions in history: {data['history_count']}")
            else:
                st.error("❌ API returned an error")
        except requests.exceptions.RequestException:
            st.error("❌ Cannot connect to API. Make sure the FastAPI server is running.")
    
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Make sure PDFs are in the `/data` folder
    2. Run the data pipeline to process PDFs
    3. Click "Generate Questions" to create unique questions
    """)

# Main content
col1, col2 = st.columns([3, 1])

with col2:
    if st.button("🚀 Generate Questions", type="primary", use_container_width=True):
        with st.spinner("Generating questions..."):
            try:
                response = requests.get(
                    f"{API_URL}/generate",
                    params={"count": question_count},
                    timeout=300  # 5 minutes timeout for generation
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state['questions'] = data['questions']
                    st.session_state['generation_info'] = {
                        'count': data['count'],
                        'total_in_history': data['total_in_history']
                    }
                    st.success(f"✅ Generated {data['count']} questions!")
                else:
                    st.error(f"❌ Error: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Cannot connect to API: {str(e)}")

# Display questions
if 'questions' in st.session_state and st.session_state['questions']:
    questions = st.session_state['questions']
    info = st.session_state.get('generation_info', {})
    
    st.markdown("---")
    st.subheader(f"Generated Questions ({info.get('count', len(questions))})")
    st.caption(f"Total questions in history: {info.get('total_in_history', 0)}")
    
    for idx, question in enumerate(questions, 1):
        with st.expander(f"Question {idx}: {question.get('topic', 'Unknown Topic')} - {question.get('difficulty', 'Medium')}", expanded=False):
            col_a, col_b = st.columns([3, 1])
            
            with col_a:
                st.markdown(f"**Subject:** {question.get('subject', 'N/A')}")
                st.markdown(f"**Topic:** {question.get('topic', 'N/A')}")
                st.markdown(f"**Difficulty:** {question.get('difficulty', 'N/A')}")
            
            with col_b:
                st.markdown(f"**ID:** `{question.get('id', 'N/A')[:8]}...`")
            
            st.markdown("---")
            st.markdown(f"### {question.get('question', 'No question text')}")
            
            # Display options if available
            options = question.get('options', {})
            if options:
                st.markdown("**Options:**")
                for option_key, option_value in sorted(options.items()):
                    is_correct = option_key == question.get('correct_answer', '')
                    marker = "✅" if is_correct else ""
                    st.markdown(f"{marker} **{option_key}:** {option_value}")
            
            # Display explanation
            explanation = question.get('explanation', '')
            if explanation:
                with st.expander("View Explanation"):
                    st.markdown(explanation)
            
            # Display source documents
            sources = question.get('source_documents', [])
            if sources:
                with st.expander("View Sources"):
                    for source in sources:
                        st.markdown(f"- **File:** {source.get('filename', 'Unknown')}")
                        st.markdown(f"  - Source: {source.get('source', 'Unknown')}")
            
            st.markdown("---")
    
    # Download button
    st.download_button(
        label="📥 Download Questions as JSON",
        data=json.dumps(questions, indent=2),
        file_name=f"questions_{len(questions)}.json",
        mime="application/json"
    )

else:
    st.info("👆 Click 'Generate Questions' to start generating questions from your PDFs")

# Footer
st.markdown("---")
st.markdown("### 📝 Notes")
st.markdown("""
- Questions are generated using RAG (Retrieval-Augmented Generation)
- Each question is unique and tracked to avoid duplicates
- Make sure your PDFs are processed before generating questions
- Run `python pipeline/load_data.py` to process PDFs in the `/data` folder
""")

