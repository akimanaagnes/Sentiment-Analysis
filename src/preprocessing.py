
import pandas as pd
import re

# 1. Load the original dataset
file_path = "dataset/Cell_Phones_and_Accessories_5.json"

print("Loading dataset...")
df = pd.read_json(file_path, lines=True)

# 2. Keep only the columns we need
df = df[["reviewText", "overall"]].copy()

# 3. Remove missing and empty reviews
#df.dropna(subset=["reviewText", "overall"], inplace=True)
df = df.dropna(subset=["reviewText", "overall"])
df["reviewText"] = df["reviewText"].astype(str).str.strip()
df = df[df["reviewText"] != ""]

# 4. Remove duplicate reviews
#df.drop_duplicates(subset=["reviewText", "overall"], inplace=True)
df = df.drop_duplicates(subset=["reviewText", "overall"])
# 5. Create sentiment labels
def get_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"

df["sentiment"] = df["overall"].apply(get_sentiment)

# 6. Clean review text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["cleaned_text"] = df["reviewText"].apply(clean_text)

# 7. Remove reviews that became empty
df = df[df["cleaned_text"] != ""]

# 8. Save the processed dataset
df.to_csv("dataset/processed_reviews.csv", index=False)

# 9. Display results
print("\nPreprocessing completed!")
print("\nDataset shape:", df.shape)

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nExample of cleaned reviews:")
print(df[["reviewText", "cleaned_text", "sentiment"]].head())

print("\nProcessed dataset saved successfully!")
