
import pandas as pd
import matplotlib.pyplot as plt

# Load processed dataset
df = pd.read_csv("dataset/processed_reviews.csv")

print("Dataset loaded successfully!")

# Display dataset information
print("\nDataset information:")
print(df.info())

# Display sample reviews
print("\nSample reviews:")
for index, row in df.head(5).iterrows():
    print("\nReview:", row["cleaned_text"])
    print("Rating:", row["overall"])
    print("Sentiment:", row["sentiment"])
    print("-" * 60)

# Calculate review length
df["review_length"] = df["cleaned_text"].apply(len)

print("\nAverage review length:")
print(df["review_length"].mean())

# Plot sentiment distribution
df["sentiment"].value_counts().plot(
    kind="bar",
    title="Distribution of Customer Sentiments",
    xlabel="Sentiment",
    ylabel="Number of Reviews"
)

plt.tight_layout()
plt.savefig("outputs/sentiment_distribution.png")
# plt.show()
plt.close()

plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

print("\nExploration completed!")
