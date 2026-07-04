import os
import pickle
import re
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained Naive Bayes model and TF-IDF vectorizer
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(
        "Model or vectorizer file is missing. Please run train_model.py first."
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

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

# --- Routing ---

@app.route("/")
def home():
    """Renders the Home page."""
    return render_template("index.html")

@app.route("/detector")
def detector():
    """Renders the Spam Detector interface."""
    return render_template("detector.html")

@app.route("/about")
def about():
    """Renders the About page."""
    return render_template("about.html")

@app.route("/contact")
def contact():
    """Renders the Contact page."""
    return render_template("contact.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles JSON POST requests to classify a message as Spam or Not Spam.
    Expects JSON body: { "text": "Your message here" }
    Returns JSON response with prediction and confidence.
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text content provided"}), 400

    raw_text = data["text"]
    if not raw_text.strip():
        return jsonify({"error": "Message cannot be empty"}), 400

    # 1. Clean the text using the exact same preprocessing logic as training
    cleaned = clean_text(raw_text)

    # 2. Transform using the loaded TF-IDF vectorizer
    vectorized = vectorizer.transform([cleaned])

    # 3. Perform prediction
    prediction = model.predict(vectorized)[0]  # Returns 'ham' or 'spam'
    
    # 4. Get confidence probability
    probabilities = model.predict_proba(vectorized)[0]
    classes = list(model.classes_)  # e.g., ['ham', 'spam']
    pred_idx = classes.index(prediction)
    confidence = probabilities[pred_idx] * 100

    # 5. Map internal labels ('ham', 'spam') to friendly display text
    label_map = {
        "ham": "Not Spam",
        "spam": "Spam"
    }
    result = label_map.get(prediction, "Unknown")

    return jsonify({
        "prediction": result,
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)
