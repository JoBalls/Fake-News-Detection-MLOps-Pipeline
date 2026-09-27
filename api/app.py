from flask import Flask, request, jsonify
import joblib
import re
import string
import os

app = Flask(__name__)

MODEL_PATH = os.path.join('models', 'model.pkl')
VECTORIZER_PATH = os.path.join('data', 'processed', 'tfidf_vectorizer.pkl')

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)

    if not data or 'text' not in data:
        return jsonify({"error": "The 'text' field is required in the JSON body."}), 400

    text = data['text']
    if not isinstance(text, str) or text.strip() == "":
        return jsonify({"error": "The 'text' field cannot be empty."}), 400

    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]

    label = "real" if prediction == 1 else "fake"
    confidence = float(max(probability))

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 4)
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)