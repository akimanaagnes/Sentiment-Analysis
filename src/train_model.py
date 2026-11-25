
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    f1_score
)

# Create output folders if they do not exist
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# 1. Load processed dataset
print("Loading processed dataset...")
df = pd.read_csv("dataset/processed_reviews.csv")

X = df["cleaned_text"]
y = df["sentiment"]

# 2. Split data into training and testing sets
print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training reviews:", len(X_train))
print("Testing reviews:", len(X_test))

# 3. Convert text into numerical features using TF-IDF
print("\nApplying TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

# Fit only on training data to prevent data leakage
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF completed!")
print("Training matrix shape:", X_train_tfidf.shape)
print("Testing matrix shape:", X_test_tfidf.shape)

# 4. Define machine learning models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        random_state=42
    ),

    "Multinomial Naive Bayes": MultinomialNB(
        alpha=1.0
    )
}

results = []
predictions = {}

# 5. Train and evaluate each model
for name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    predictions[name] = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    print("\nAccuracy:", round(accuracy, 4))
    print("Macro F1-score:", round(macro_f1, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1-score": macro_f1
    })

# 6. Compare models
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

# 7. Select the best model using macro F1-score
best_model_name = results_df.loc[
    results_df["Macro F1-score"].idxmax(),
    "Model"
]

best_model = models[best_model_name]

print("\nBest model:", best_model_name)

# 8. Save the best model and TF-IDF vectorizer
joblib.dump(best_model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nBest model saved successfully!")
print("TF-IDF vectorizer saved successfully!")

# 9. Create confusion matrix for the best model
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions[best_model_name],
    labels=["negative", "neutral", "positive"],
    cmap="Blues",
    xticks_rotation="vertical"
)

plt.title("Confusion Matrix - " + best_model_name)
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.show()

print("\nModel training and evaluation completed!")
