import pandas as pd
import re

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only needed columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 🔥 Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    return text

df['message'] = df['message'].apply(clean_text)

# 🔥 Balance dataset
spam_df = df[df['label'] == 1]
ham_df = df[df['label'] == 0].sample(len(spam_df), random_state=42)
df = pd.concat([spam_df, ham_df])

# 🔥 Better vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2))

X = vectorizer.fit_transform(df['message'])
y = df['label']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 Better model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Prediction function
def predict_spam(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]
    return "Spam" if result == 1 else "Not Spam"

# Interactive input
while True:
    msg = input("\nEnter message (or type 'exit'): ")
    if msg.lower() == 'exit':
        break
    print("Prediction:", predict_spam(msg))