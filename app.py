#!/usr/bin/env python3
"""
Cyberbullying Detection Web Application
User pastes tweet → Preprocessing → TF-IDF → SVM Classification → Result
"""

from flask import Flask, render_template, request, jsonify
import pickle
import re
import json
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import numpy as np

app = Flask(__name__)

# ─── LOAD MODELS & VECTORIZER ──────────────────────────────────────────────────
print("Loading trained model and vectorizer...")
with open('best_cyberbullying_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

# Initialize Indonesian NLP tools
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()
sw_factory = StopWordRemoverFactory()
stopwords = set(sw_factory.get_stop_words())

print("✓ Model loaded successfully!")
print(f"✓ Model: {metadata['model_name']}")
print(f"✓ Accuracy: {metadata['metrics']['accuracy']:.1%}")
print(f"✓ Categories: {len(metadata['categories'])}")

# ─── PREPROCESSING FUNCTION ────────────────────────────────────────────────────
def preprocess_tweet(text):
    """Apply all 5 preprocessing steps to raw tweet"""
    
    # Step 1: Text Cleaning
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)        # URLs
    text = re.sub(r'@\w+', '', text)                          # @mentions
    text = re.sub(r'#\w+', '', text)                          # #hashtags
    text = re.sub(r'<[^>]+>', '', text)                       # HTML tags
    text = re.sub(r'RT\s*:', '', text)                        # RT markers
    text = re.sub(r'[^\w\s]', '', text)                       # Special chars
    text = re.sub(r'\d+', '', text)                           # Numbers
    text = re.sub(r'\s+', ' ', text).strip()                  # Whitespace
    
    # Step 2: Case Folding
    text = text.lower()
    
    # Step 3: Tokenization
    tokens = text.split()
    
    # Step 4: Stopword Removal
    tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
    
    # Step 5: Stemming
    tokens = [stemmer.stem(t) for t in tokens]
    
    return ' '.join(tokens)

# Minimum confidence to classify as cyberbullying (below this = "Bukan Cyberbullying")
CONFIDENCE_THRESHOLD = 0.40

# ─── CATEGORY DESCRIPTIONS ────────────────────────────────────────────────────
category_descriptions = {
    'Umpatan Kasar & Binatang': 'Harsh words and animal insults directed at someone',
    'Body Shaming & Kondisi Fisik': 'Negative comments about physical appearance or body condition',
    'Degradasi Moral & Pelecehan Seksual': 'Moral degradation and sexual harassment',
    'Serangan Psikologis & Sindiran': 'Psychological attacks, mockery, and sarcasm',
    'Merendahkan Intelektual & Sosial': 'Intellectual and social degradation',
    'Politik / SARA / Labeling': 'Political labels, SARA (sensitive identity issues), and stereotyping',
    'Bukan Cyberbullying': 'This text does not appear to contain cyberbullying content'
}

# Color codes for categories (for UI)
category_colors = {
    'Umpatan Kasar & Binatang': '#FF6B6B',
    'Body Shaming & Kondisi Fisik': '#FF8C42',
    'Degradasi Moral & Pelecehan Seksual': '#D64A3E',
    'Serangan Psikologis & Sindiran': '#F77F88',
    'Merendahkan Intelektual & Sosial': '#FFAF60',
    'Politik / SARA / Labeling': '#C44536',
    'Bukan Cyberbullying': '#4CAF50'
}

# ─── ROUTES ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Homepage with classification form"""
    return render_template('index.html', categories=metadata['categories'])

@app.route('/api/classify', methods=['POST'])
def classify():
    """API endpoint for tweet classification"""
    try:
        data = request.json
        tweet = data.get('tweet', '').strip()
        
        if not tweet:
            return jsonify({'error': 'Please enter a tweet'}), 400
        
        if len(tweet) < 5:
            return jsonify({'error': 'Tweet too short (minimum 5 characters)'}), 400
        
        # Preprocess the tweet
        preprocessed = preprocess_tweet(tweet)
        
        if not preprocessed.strip():
            return jsonify({'error': 'Tweet text is empty after preprocessing'}), 400
        
        # Vectorize using saved TF-IDF
        X = vectorizer.transform([preprocessed])

        # Get raw SVM decision scores for all classes
        decision_scores = model.decision_function(X)[0]

        # Softmax-normalize scores to get probabilities
        scores_shifted = decision_scores - np.max(decision_scores)
        exp_scores = np.exp(scores_shifted)
        probabilities = exp_scores / np.sum(exp_scores)

        # Best class index and confidence
        best_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[best_idx])
        predicted_class = model.classes_[best_idx]

        # Apply threshold: if not confident enough → Not Cyberbullying
        if confidence < CONFIDENCE_THRESHOLD:
            prediction = 'Bukan Cyberbullying'
        else:
            prediction = predicted_class

        return jsonify({
            'success': True,
            'original_tweet': tweet,
            'preprocessed_text': preprocessed,
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'description': category_descriptions.get(prediction, 'Cyberbullying detected'),
            'color': category_colors.get(prediction, '#999999'),
            'model': metadata['model_name'],
            'accuracy': metadata['metrics']['accuracy']
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/info')
def info():
    """Return model information"""
    return jsonify({
        'model_name': metadata['model_name'],
        'accuracy': metadata['metrics']['accuracy'],
        'precision': metadata['metrics']['precision'],
        'recall': metadata['metrics']['recall'],
        'f1_score': metadata['metrics']['f1_score'],
        'categories': metadata['categories'],
        'total_training_samples': 1628,
        'total_test_samples': 408,
        'feature_extraction': metadata['feature_extraction'],
        'all_models': metadata['all_models_comparison']
    })

# ─── ERROR HANDLERS ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("CYBERBULLYING DETECTION WEB APPLICATION")
    print("=" * 70)
    print("\n✓ Flask app running!")
    print("✓ Visit: http://localhost:5000")
    print("✓ API endpoint: POST /api/classify")
    print("=" * 70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
