
import streamlit as st
import pandas as pd
import joblib
import re
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Amazon Sentiment Intelligence",
    page_icon="💬",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 28px;
        border-radius: 15px;
        background: linear-gradient(120deg, #14213d, #2563eb);
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        font-size: 32px;
    }

    .hero p {
        color: #e2e8f0;
        font-size: 16px;
    }

    .small-note {
        color: #64748b;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_models():
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_models()


# --------------------------------------------------
# TEXT PREPROCESSING
# --------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("💬 Sentiment AI")
st.sidebar.caption("Amazon Customer Review Analysis")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Home / Dashboard",
        "Sentiment Prediction",
        "Model Performance",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Machine Learning | NLP | Streamlit")
st.sidebar.caption("Academic Data Science Project")


# --------------------------------------------------
# HOME / DASHBOARD
# --------------------------------------------------

if page == "Home / Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>Amazon Review Sentiment Analyzer</h1>
        <p>
        An intelligent system that uses Natural Language Processing
        and Machine Learning to understand customer opinions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Reviews", "194,210")
    col2.metric("Sentiment Classes", "3")
    col3.metric("Model Accuracy", "78.79%")
    col4.metric("Macro F1-score", "65.76%")

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Project Introduction")

        st.write("""
        This project analyzes Amazon customer reviews and classifies
        them into three sentiment categories:

        - 🟢 **Positive:** Satisfied customer opinions.
        - 🔴 **Negative:** Dissatisfied customer opinions.
        - 🟡 **Neutral:** Mixed or neither clearly positive nor negative.

        The system uses TF-IDF for text representation and
        Logistic Regression for sentiment classification.
        """)

    with right:
        st.subheader("Sentiment Distribution")

        sentiment_data = pd.DataFrame({
            "Sentiment": ["Positive", "Negative", "Neutral"],
            "Reviews": [148450, 24327, 21433]
        })

        st.bar_chart(
            sentiment_data.set_index("Sentiment")
        )

    st.divider()

    st.subheader("How the System Works")

    step1, step2, step3 = st.columns(3)

    step1.markdown("### 1. Input")
    step1.write("Enter a customer review.")

    step2.markdown("### 2. Processing")
    step2.write("Text is cleaned and transformed using TF-IDF.")

    step3.markdown("### 3. Prediction")
    step3.write("The trained model predicts the sentiment.")


# --------------------------------------------------
# SENTIMENT PREDICTION
# --------------------------------------------------

elif page == "Sentiment Prediction":

    st.title("🔍 Sentiment Prediction")

    st.write(
        "Enter an Amazon customer review to analyze its sentiment."
    )

    sample_review = (
        "This phone charger is amazing. "
        "It works perfectly and I love it!"
    )

    review = st.text_area(
        "Customer Review",
        value=sample_review,
        height=160,
        placeholder="Write or paste a customer review here..."
    )

    if st.button("Analyze Sentiment", type="primary"):

        if not review.strip():
            st.warning("Please enter a review first.")

        else:
            cleaned_review = clean_text(review)

            if not cleaned_review:
                st.warning(
                    "Please enter a review containing English letters."
                )

            else:
                review_vector = vectorizer.transform([cleaned_review])

                prediction = model.predict(review_vector)[0]
                probabilities = model.predict_proba(review_vector)[0]
                classes = model.classes_

                st.divider()
                st.subheader("Prediction Result")

                confidence = max(probabilities) * 100

                if prediction == "positive":
                    st.success("😊 Positive Sentiment")

                elif prediction == "negative":
                    st.error("😞 Negative Sentiment")

                else:
                    st.warning("😐 Neutral Sentiment")

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

                st.caption(
                    "Confidence is the model's estimated probability, "
                    "not a guarantee that the prediction is correct."
                )

                st.subheader("Sentiment Probabilities")

                probability_df = pd.DataFrame({
                    "Sentiment": classes,
                    "Probability": probabilities
                })

                probability_df["Probability"] *= 100

                st.bar_chart(
                    probability_df.set_index("Sentiment")
                )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("📊 Model Performance")

    st.write(
        "Evaluation results from the held-out test dataset."
    )

    col1, col2 = st.columns(2)

    col1.metric("Accuracy", "78.79%")
    col2.metric("Macro F1-score", "65.76%")

    st.divider()

    st.subheader("Machine Learning Model Comparison")

    comparison_path = "outputs/model_comparison.csv"

    if os.path.exists(comparison_path):

        results = pd.read_csv(comparison_path)

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("Model comparison results are unavailable.")

    st.divider()

    st.subheader("Confusion Matrix")

    matrix_path = "outputs/confusion_matrix.png"

    if os.path.exists(matrix_path):

        st.image(
            matrix_path,
            caption="Logistic Regression Confusion Matrix",
            use_container_width=True
        )

    else:
        st.info("Confusion matrix image is unavailable.")

    st.subheader("Model Interpretation")

    st.write("""
    Logistic Regression was selected because it achieved a higher
    Macro F1-score than Multinomial Naive Bayes in the original
    model comparison.

    Neutral sentiment remains more difficult to classify than
    positive sentiment. This is an important limitation of the model.
    """)


# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------

elif page == "About Project":

    st.title("ℹ️ About This Project")

    st.subheader("Amazon Review Sentiment Analysis")

    st.write("""
    This academic project applies Natural Language Processing
    and Machine Learning to Amazon customer reviews.

    Its main objective is to automatically identify the sentiment
    expressed in customer review text.
    """)

    st.divider()

    st.subheader("Dataset Information")

    st.write("""
    - **Source:** Kaggle – Amazon Reviews
    - **Category:** Cell Phones and Accessories
    - **Original records:** 194,439
    - **Processed records:** 194,210
    - **Text feature:** Review text
    - **Target variable:** Sentiment
    """)

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Natural Language Processing
    - TF-IDF Vectorization
    - Logistic Regression
    - Streamlit
    """)

    st.subheader("Sentiment Classification")

    st.write("""
    - 1–2 stars: Negative
    - 3 stars: Neutral
    - 4–5 stars: Positive

    These sentiment labels were derived from the original star ratings.
    They were not manually annotated.
    """)

    st.divider()

    st.subheader("Project Limitations")

    st.write("""
    The dataset contains more positive reviews than negative and
    neutral reviews. This class imbalance affects model performance.

    In addition, a three-star review can contain both positive and
    negative opinions, making neutral classification challenging.
    """)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    "<p class='small-note'>"
    "Amazon Review Sentiment Analysis | "
    "Natural Language Processing | Machine Learning | Streamlit"
    "</p>",
    unsafe_allow_html=True
)
