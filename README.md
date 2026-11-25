# Amazon Review Sentiment Analysis

## 1. Project Overview

Amazon Review Sentiment Analysis is a Machine Learning project that analyzes customer reviews and classifies them into three sentiment categories:

- Positive
- Negative
- Neutral

The project uses Natural Language Processing (NLP), TF-IDF feature extraction, and Machine Learning algorithms to predict the sentiment expressed in customer reviews.

A Streamlit web application allows users to enter a review and receive an instant sentiment prediction.

## 2. Dataset

The project uses the Amazon Reviews dataset for the Cell Phones & Accessories category, obtained from Kaggle.

- Original records: 194,439
- Processed records: 194,210
- Input feature: Review text (`reviewText`)
- Target source: Customer rating (`overall`)

### Sentiment Labeling

The sentiment labels were generated from the star ratings:

| Rating | Sentiment |
|---|---|
| 1–2 stars | Negative |
| 3 stars | Neutral |
| 4–5 stars | Positive |

**Note:** The dataset is not included in this public GitHub repository. Download the original dataset from Kaggle and place it inside the `dataset/` folder before running the preprocessing scripts.

## 3. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Matplotlib and Seaborn
- Streamlit
- Joblib

## 4. Project Methodology

1. Dataset loading and inspection.
2. Data cleaning and preprocessing.
3. Sentiment labeling based on customer ratings.
4. Exploratory Data Analysis (EDA).
5. Text feature extraction using TF-IDF.
6. Training Logistic Regression and Multinomial Naive Bayes.
7. Model comparison and evaluation.
8. Saving the best model.
9. Developing an interactive Streamlit application.

## 5. Machine Learning Models

Two algorithms were trained and compared:

- Logistic Regression
- Multinomial Naive Bayes

Logistic Regression was selected as the final model based on its higher Macro F1-score.

### Model Evaluation Results

| Model | Accuracy | Macro F1-score |
|---|---:|---:|
| Logistic Regression | 78.79% | 65.76% |
| Multinomial Naive Bayes | 81.83% | 53.09% |

## 6. Project Structure

```text
Sentiment-Analysis/
│
├── src/
│   ├── __init__.py
│   ├── inspect_dataset.py
│   ├── preprocessing.py
│   ├── explore_data.py
│   └── train_model.py
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── outputs/
│   ├── sentiment_distribution.png
│   ├── confusion_matrix.png
│   └── model_comparison.csv
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

The `dataset/` folder is used locally for the original and processed datasets but is excluded from the public repository.

## 7. Installation and Usage

### Clone the repository

```bash
git clone https://github.com/akimanaagnes/Sentiment-Analysis.git
cd Sentiment-Analysis
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment on Windows

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

```bash
streamlit run app.py
```

**Note:** The trained model and vectorizer are already included in the repository, so you do not need to retrain the model to run the application.

## 8. Application Features

- Enter customer reviews.
- Predict positive, negative, or neutral sentiment.
- Display prediction confidence.
- Visualize sentiment probabilities.
- Compare trained machine learning models.
- Display the model confusion matrix.

## 9. Model Performance

Logistic Regression achieved 78.79% accuracy and a Macro F1-score of 65.76% on the held-out test set.

The model performed best on positive reviews, while neutral sentiment was more challenging due to class imbalance and overlapping language patterns.

## 10. Conclusion

This project demonstrates how Natural Language Processing and Machine Learning can be used to analyze customer opinions and classify product reviews.

The Streamlit application provides an interactive interface for testing sentiment predictions.

## Author

Data Science Student

## Note on Dataset

The original Amazon review dataset is used for academic and educational purposes. Refer to the original dataset source and applicable license and terms before redistributing the data.
