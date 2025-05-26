import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the Dataset
df = pd.read_csv("Tweets.csv")  # Ensure this CSV is in the same folder
df = df[['text', 'airline_sentiment']]
df = df[df['airline_sentiment'].isin(['positive', 'negative'])]  # Only positive/negative
df['label'] = df['airline_sentiment'].map({'positive': 1, 'negative': 0})

# Step 2: Clean the Tweets
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-zA-Z']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Step 3: Generate Word Cloud (positive tweets)
positive_text = ' '.join(df[df['label'] == 1]['clean_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Positive Tweets")
plt.tight_layout()
plt.savefig("wordcloud_positive.png")  # Save image for Word doc
plt.show()

# Step 4: TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['clean_text'])
y = df['label']

# Step 5: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n", report)

# Step 8: Test on Sample Tweets
sample_texts = [
    "I had a great flight experience with friendly crew!",
    "My flight was delayed for hours. Terrible service!"
]
sample_cleaned = [clean_text(text) for text in sample_texts]
sample_vectorized = tfidf.transform(sample_cleaned)
sample_preds = model.predict(sample_vectorized)

print("\nSample Tweet Predictions:")
for text, pred in zip(sample_texts, sample_preds):
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"- {text} --> {sentiment}")
