import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("best_spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit app title
st.title("🔒 Spam Classifier")

# Text area for user input
email_text = st.text_area("Enter Email Text", placeholder="Paste your email here...")

# Classify button
if st.button("Classify Email"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter email text to classify.")
    else:
        # Clean and transform the input text
        # Assuming clean_text function is available or defined within app.py if needed.
        # For this example, let's assume it's part of the pre-processing chain handled by the vectorizer or model.
        # If clean_text was a separate function, it would need to be included here.

        # For now, directly transform. In a real deployment, ensure clean_text is applied.
        transformed = vectorizer.transform([email_text])
        
        # Make prediction
        prediction = model.predict(transformed)
        confidence = model.predict_proba(transformed).max()

        # Display results
        if prediction[0] == 1:
            st.error(f"🚨 **Spam Detected** (Confidence: {confidence:.2%})")
        else:
            st.success(f"✅ **Legitimate Email** (Confidence: {confidence:.2%})")

        st.markdown("---")
        st.info(f"Classification Confidence: {confidence:.2%}")
