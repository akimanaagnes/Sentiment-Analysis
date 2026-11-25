
import streamlit as st
import pandas as pd
import joblib
import re
import os

# Page configuration
st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="💬",
    layout="wide"
)

# Load trained model and vectorizer
@st.cache_resource
def load_models():
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()

# Clean input text exactly as during preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Application heading
st.title("💬 Amazon Review Sentiment Analyzer")

st.write(
    "Analyze customer reviews using Machine Learning "
    "and predict whether they express positive, negative, "
    "or neutral sentiment."
)

st.divider()

# Main prediction section
st.header("Analyze a Customer Review")

sample_review = (
    "This phone charger is amazing. "
    "It works perfectly and I love it!"
)

review = st.text_area(
    "Enter your customer review:",
    value=sample_review,
    height=150
)

if st.button("Analyze Sentiment", type="primary"):

    if not review.strip():
        st.warning("Please enter a review first.")

    else:
        cleaned_review = clean_text(review)

        if not cleaned_review:
            st.warning("Please enter a review containing English letters.")

        else:
            # Convert text into TF-IDF features
            review_vector = vectorizer.transform([cleaned_review])

            # Predict sentiment
            prediction = model.predict(review_vector)[0]

            # Get prediction probabilities
            probabilities = model.predict_proba(review_vector)[0]

            classes = model.classes_

            st.subheader("Prediction Result")

            if prediction == "positive":
                st.success("😊 Positive Sentiment")

            elif prediction == "negative":
                st.error("😞 Negative Sentiment")

            else:
                st.warning("😐 Neutral Sentiment")

            # Display confidence
            confidence = max(probabilities) * 100

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            # Display probabilities for all classes
            st.subheader("Sentiment Probabilities")

            probability_df = pd.DataFrame({
                "Sentiment": classes,
                "Probability": probabilities
            })

            st.bar_chart(
                probability_df.set_index("Sentiment")
            )

st.divider()

# Model comparison section
st.header("Machine Learning Model Comparison")

if os.path.exists("outputs/model_comparison.csv"):

    results = pd.read_csv("outputs/model_comparison.csv")

    st.dataframe(
        results.style.format({
            "Accuracy": "{:.2%}",
            "Macro F1-score": "{:.2%}"
        }),
        use_container_width=True
    )

else:
    st.info("Model comparison results are not available.")

# Confusion matrix section
st.header("Model Evaluation")

if os.path.exists("outputs/confusion_matrix.png"):
    st.image(
        "outputs/confusion_matrix.png",
        caption="Logistic Regression Confusion Matrix",
        width=600
    )

# Footer
st.divider()

st.caption(
    "Sentiment Analysis Project | Amazon Customer Reviews | "
    "Machine Learning with TF-IDF and Logistic Regression"
)
