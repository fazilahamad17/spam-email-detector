import os
import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def clean_text(text):
    # Ensure text is string
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train():
    print("Step 1: Loading dataset...")
    dataset_path = os.path.join("dataset", "spam.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    df = pd.read_csv(dataset_path)
    
    # Check dataset columns
    if 'label' not in df.columns or 'text' not in df.columns:
        raise ValueError("Dataset must contain 'label' and 'text' columns")
        
    print(f"Loaded {len(df)} samples.")

    print("\nStep 2: Cleaning text data...")
    df['clean_text'] = df['text'].apply(clean_text)

    # Split dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], 
        df['label'], 
        test_size=0.2, 
        random_state=42
    )

    print("\nStep 3: Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    print("\nStep 4: Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)

    # Evaluate the model
    predictions = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nModel training complete!")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nStep 5: Saving model and vectorizer...")
    with open("model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
        
    with open("vectorizer.pkl", "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)
        
    print("Files saved successfully: model.pkl, vectorizer.pkl")

if __name__ == "__main__":
    train()
