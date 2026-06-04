import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 50px;
        font-size: 16px;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .spam-badge {
        background-color: #ff4444;
        color: white;
        padding: 10px 15px;
        border-radius: 5px;
        font-weight: bold;
    }
    .legitimate-badge {
        background-color: #44ff44;
        color: black;
        padding: 10px 15px;
        border-radius: 5px;
        font-weight: bold;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("best_spam_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Sidebar Configuration
with st.sidebar:
    st.markdown("## ⚙️ Settings & Information")
    st.markdown("---")
    
    # Theme selection
    theme = st.selectbox("🎨 Select Theme", ["Light", "Dark", "Auto"])
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "🎯 Confidence Threshold",
        0.0, 1.0, 0.5,
        help="Minimum confidence required to classify as spam"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Type", "SVM")
    with col2:
        st.metric("Vectorizer", "TF-IDF")
    
    st.info("✅ Model loaded successfully and ready for predictions")
    
    st.markdown("---")
    st.markdown("### 📧 Quick Stats")
    if 'classification_history' in st.session_state:
        total = len(st.session_state.classification_history)
        spam_count = sum(1 for item in st.session_state.classification_history if item['result'] == 'Spam')
        st.write(f"**Emails Classified:** {total}")
        st.write(f"**Spam Detected:** {spam_count}")
    else:
        st.write("**Emails Classified:** 0")
        st.write("**Spam Detected:** 0")

# Initialize session state for history
if 'classification_history' not in st.session_state:
    st.session_state.classification_history = []

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Classifier", "📊 Batch Upload", "📈 Analytics", "ℹ️ About"])

# TAB 1: Single Email Classifier
with tab1:
    st.markdown("## 🔒 Email Spam Classifier")
    st.markdown("Enter an email text below to check if it's spam or legitimate")
    
    # Create columns for better layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        email_text = st.text_area(
            "Enter Email Text",
            placeholder="Paste your email here...",
            height=250,
            max_chars=5000,
            label_visibility="collapsed"
        )
        st.write(f"Characters: {len(email_text)}/5000")
    
    with col2:
        st.markdown("### 📋 Tips")
        st.markdown("""
        - Include subject line
        - Include body text
        - Include sender info
        - Check for urgency language
        """)
    
    # Classify button
    if st.button("🚀 Classify Email", type="primary"):
        if email_text.strip() == "":
            st.warning("⚠️ Please enter email text to classify.")
        else:
            with st.spinner("Analyzing email..."):
                # Transform the input text
                transformed = vectorizer.transform([email_text])
                
                # Make prediction
                prediction = model.predict(transformed)
                confidence = model.predict_proba(transformed).max()
                
                # Store in history
                st.session_state.classification_history.append({
                    'timestamp': datetime.now(),
                    'email_preview': email_text[:100] + "...",
                    'result': 'Spam' if prediction[0] == 1 else 'Legitimate',
                    'confidence': confidence
                })
                
                # Display results with improved styling
                st.markdown("---")
                st.markdown("## 📊 Classification Result")
                
                result_col1, result_col2, result_col3 = st.columns(3)
                
                if prediction[0] == 1:
                    with result_col1:
                        st.markdown('<div class="spam-badge">🚨 SPAM DETECTED</div>', unsafe_allow_html=True)
                    with result_col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    with result_col3:
                        st.metric("Risk Level", "🔴 High")
                else:
                    with result_col1:
                        st.markdown('<div class="legitimate-badge">✅ LEGITIMATE</div>', unsafe_allow_html=True)
                    with result_col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    with result_col3:
                        st.metric("Risk Level", "🟢 Low")
                
                st.markdown("---")
                
                # Additional information
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.markdown("### 📈 Prediction Details")
                    st.json({
                        "Classification": "Spam" if prediction[0] == 1 else "Legitimate",
                        "Confidence": f"{confidence:.2%}",
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                
                with info_col2:
                    # Gauge chart for confidence
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=confidence * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Confidence Score"},
                        delta={'reference': confidence_threshold * 100},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgray"},
                                {'range': [25, 50], 'color': "gray"},
                                {'range': [50, 75], 'color': "lightblue"},
                                {'range': [75, 100], 'color': "blue"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': confidence_threshold * 100
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommendation
                st.markdown("---")
                if prediction[0] == 1 and confidence >= confidence_threshold:
                    st.error("🛑 Recommendation: DELETE or REPORT this email as spam")
                elif prediction[0] == 1:
                    st.warning("⚠️ Recommendation: REVIEW this email carefully before interacting")
                else:
                    st.success("✓ Recommendation: This email appears safe to read")

# TAB 2: Batch Upload
with tab2:
    st.markdown("## 📊 Batch Email Classification")
    st.markdown("Upload a CSV file with emails to classify multiple messages at once")
    
    st.info("📝 CSV Format: Your file should have a column named 'email' or 'text' containing email content")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Find email column
        email_column = None
        for col in ['email', 'text', 'Email', 'Text', 'EMAIL', 'TEXT']:
            if col in df.columns:
                email_column = col
                break
        
        if email_column is None:
            st.error("❌ CSV must contain a column named 'email' or 'text'")
        else:
            st.write(f"Found {len(df)} emails to classify")
            
            if st.button("🚀 Classify All Emails"):
                progress_bar = st.progress(0)
                results = []
                
                for idx, row in df.iterrows():
                    email_text = row[email_column]
                    if pd.notna(email_text):
                        transformed = vectorizer.transform([str(email_text)])
                        prediction = model.predict(transformed)
                        confidence = model.predict_proba(transformed).max()
                        
                        results.append({
                            'email_preview': str(email_text)[:100],
                            'classification': 'Spam' if prediction[0] == 1 else 'Legitimate',
                            'confidence': confidence
                        })
                    progress_bar.progress((idx + 1) / len(df))
                
                results_df = pd.DataFrame(results)
                
                # Display statistics
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Emails", len(results_df))
                with col2:
                    spam_count = len(results_df[results_df['classification'] == 'Spam'])
                    st.metric("Spam Detected", spam_count)
                with col3:
                    legitimate_count = len(results_df[results_df['classification'] == 'Legitimate'])
                    st.metric("Legitimate", legitimate_count)
                with col4:
                    avg_confidence = results_df['confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                
                st.markdown("---")
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name=f"spam_classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# TAB 3: Analytics
with tab3:
    st.markdown("## 📈 Analytics & Statistics")
    
    if len(st.session_state.classification_history) > 0:
        history_df = pd.DataFrame(st.session_state.classification_history)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Classifications", len(history_df))
        with col2:
            spam_count = len(history_df[history_df['result'] == 'Spam'])
            st.metric("Spam Count", spam_count)
        with col3:
            avg_confidence = history_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
        
        st.markdown("---")
        
        # Chart 1: Classification distribution
        classification_counts = history_df['result'].value_counts()
        fig = go.Figure(data=[
            go.Pie(labels=classification_counts.index, values=classification_counts.values, hole=0.3)
        ])
        fig.update_layout(title="Classification Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Chart 2: Confidence distribution
        fig2 = go.Figure(data=[
            go.Histogram(x=history_df['confidence'], nbinsx=20, name='Confidence')
        ])
        fig2.update_layout(title="Confidence Score Distribution", xaxis_title="Confidence", yaxis_title="Frequency")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Recent classifications
        st.markdown("### 📋 Recent Classifications")
        st.dataframe(history_df[['timestamp', 'result', 'confidence']].tail(10), use_container_width=True)
    else:
        st.info("📊 No classifications yet. Start by classifying emails in the 'Classifier' tab!")

# TAB 4: About
with tab4:
    st.markdown("## ℹ️ About Spam Classifier")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 What is This?
        
        The Spam Classifier is a machine learning application designed to detect 
        spam emails with high accuracy. It uses:
        
        - **Algorithm:** Support Vector Machine (SVM)
        - **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
        - **Language:** Python with Streamlit
        
        ### 📊 How It Works
        
        1. **Input Processing:** The email text is cleaned and processed
        2. **Feature Extraction:** Text is converted to numerical features using TF-IDF
        3. **Classification:** The SVM model predicts whether the email is spam
        4. **Confidence:** The model provides a confidence score for the prediction
        """)
    
    with col2:
        st.markdown("""
        ### ✨ Features
        
        - ✅ Single email classification
        - ✅ Batch processing with CSV upload
        - ✅ Real-time confidence scores
        - ✅ Classification history & analytics
        - ✅ Customizable confidence threshold
        - ✅ Download results
        
        ### 🔧 Technical Stack
        
        - **Frontend:** Streamlit
        - **ML Framework:** Scikit-learn
        - **Visualization:** Plotly
        - **Data Processing:** Pandas
        
        ### 📧 Example Spam Indicators
        
        - Urgent action required
        - Click here immediately
        - Verify account information
        - Claim your prize
        - Nigerian prince offers
        - Confirm payment details
        """)
    
    st.markdown("---")
    st.markdown("""
    ### 🚀 Getting Started
    
    1. Go to the **Classifier** tab
    2. Paste an email or email content
    3. Click **Classify Email**
    4. Review the results and confidence score
    
    For batch processing, use the **Batch Upload** tab to classify multiple emails at once.
    """)
    
    st.markdown("---")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
