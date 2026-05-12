# 🛡️ Cyberbullying Detection System

A machine learning-powered web application that detects and classifies cyberbullying in Indonesian tweets from Social Media X (Twitter).

**Status**: ✅ Complete & Ready to Deploy | **Accuracy**: 89% | **Model**: SVM

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start-5-minutes)
2. [Project Structure](#-project-structure)
3. [Installation](#-installation)
4. [Usage](#-usage)
5. [How It Works](#-how-it-works)
6. [Model Performance](#-model-performance)
7. [Deployment](#-deployment)
8. [Technical Details](#-technical-details)

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify files are in place
# Ensure these exist in your folder:
# - app.py
# - best_cyberbullying_model.pkl
# - tfidf_vectorizer.pkl
# - model_metadata.json
# - templates/index.html

# 3. Run the application
python app.py

# 4. Open browser
# Visit: http://localhost:5000
```

### Test It!

```
1. Paste a tweet in Indonesian (e.g., "lu goblok ni muka mu jelek banget")
2. Click "Classify Tweet"
3. See the cyberbullying category result (within 100ms!)
```

---

## 📁 Project Structure

```
cyberbullying-detection/
├── app.py                              # Flask web server (run this!)
├── train_models.py                     # ML training pipeline
├── preprocess.py                       # Data preprocessing script
├── templates/
│   └── index.html                      # Beautiful web interface
├── best_cyberbullying_model.pkl        # Trained SVM classifier (89% accuracy)
├── tfidf_vectorizer.pkl                # TF-IDF feature vectorizer
├── model_metadata.json                 # Model info & performance metrics
├── Merged_Preprocessed_Cyberbullying.xlsx  # Full dataset with all preprocessing steps
├── ML_Ready_Cyberbullying.csv          # Training data (ready for ML pipeline)
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## 💻 Installation

### Option 1: Windows

```bash
# 1. Open Command Prompt (Win + R, type cmd)

# 2. Navigate to your folder
cd Desktop\cyberbullying-detection

# 3. Create virtual environment
python -m venv venv

# 4. Activate it
venv\Scripts\activate

# 5. Install packages
pip install -r requirements.txt

# 6. Run app
python app.py
```

### Option 2: Mac/Linux

```bash
# 1. Open Terminal

# 2. Navigate to your folder
cd ~/Desktop/cyberbullying-detection

# 3. Create virtual environment
python3 -m venv venv

# 4. Activate it
source venv/bin/activate

# 5. Install packages
pip install -r requirements.txt

# 6. Run app
python3 app.py
```

---

## 📖 Usage

### Running the Web App

```bash
python app.py

# Expected output:
# ========================================
# CYBERBULLYING DETECTION WEB APPLICATION
# ========================================
# ✓ Flask app running!
# ✓ Visit: http://localhost:5000
# ✓ API endpoint: POST /api/classify
# ========================================
```

### Using the Interface

**Step 1**: Copy a tweet from X (Twitter)
```
Example: "lu tolol ni bodoh banget muka mu jelek"
```

**Step 2**: Paste in the textbox at `http://localhost:5000`

**Step 3**: Click "Classify Tweet" button

**Step 4**: View results showing:
- 🔴 **Category** (one of 6 cyberbullying types)
- 📊 **Confidence** (0-100%)
- 📝 **Description** (what the category means)
- 🔍 **Preprocessed Text** (how the system processed it)

---

## 🧠 How It Works

### Pipeline Overview

```
Raw Tweet
    ↓
[1] Text Cleaning (remove URLs, @, #, special chars)
    ↓
[2] Case Folding (convert to lowercase)
    ↓
[3] Tokenization (split into words)
    ↓
[4] Stopword Removal (filter 809 Indonesian stopwords)
    ↓
[5] Stemming (reduce to root forms - Sastrawi)
    ↓
[6] TF-IDF Vectorization (convert to 2,553 features)
    ↓
[7] SVM Prediction (classify into 6 categories)
    ↓
Result: Category + Confidence Score
```

### 6️⃣ Cyberbullying Categories

| # | Category | Description | Keywords |
|---|----------|-------------|----------|
| 1 | **Umpatan Kasar & Binatang** | Harsh words, animal insults | tolol, goblok, anjing, babi |
| 2 | **Body Shaming & Kondisi Fisik** | Negative comments about appearance | gendut, jelek, cacat, muka burik |
| 3 | **Degradasi Moral & Pelecehan Seksual** | Moral degradation, sexual harassment | jablay, pelacur, lonte, banci |
| 4 | **Serangan Psikologis & Sindiran** | Psychological attacks, mockery | mati aja, gak berguna, dibully |
| 5 | **Merendahkan Intelektual & Sosial** | Intellectual/social put-downs | otak udang, jamet, miskin, udik |
| 6 | **Politik / SARA / Labeling** | Political labels, stereotyping | cebong, kadrun, kafir, komunis |

---

## 📊 Model Performance

### Overall Metrics
- **Algorithm**: Support Vector Machine (SVM)
- **Test Accuracy**: 89.0%
- **Weighted Precision**: 89.1%
- **Weighted Recall**: 89.0%
- **Weighted F1-Score**: 88.9%

### Per-Category F1-Scores
```
Degradasi Moral & Pelecehan Seksual:  95.4% ⭐ (Best)
Body Shaming & Kondisi Fisik:         91.4%
Serangan Psikologis & Sindiran:       91.1%
Merendahkan Intelektual & Sosial:     85.7%
Umpatan Kasar & Binatang:             84.5%
Politik / SARA / Labeling:            84.8%
```

### Training Details
- **Dataset**: 2,036 tweets
- **Training samples**: 1,628 (80%)
- **Test samples**: 408 (20%)
- **Features**: 2,553 (TF-IDF vectorization)
- **Models compared**: Naive Bayes (77.5%), SVM (89.0%), Random Forest (70.1%)

---

## 🌐 Deployment

### Local Testing
```bash
python app.py
# Visit: http://localhost:5000
```

### Cloud Deployment (Heroku - Recommended)

```bash
# 1. Install Heroku CLI
# 2. Login
heroku login

# 3. Create app
heroku create your-app-name

# 4. Create Procfile
echo "web: python app.py" > Procfile

# 5. Deploy
git push heroku main

# 6. Visit
# https://your-app-name.herokuapp.com
```

### Other Options
- **Streamlit Cloud** (easiest for Python apps)
- **Railway** / **Render** (modern alternatives)
- **PythonAnywhere** (shared hosting)
- **AWS / Google Cloud** (enterprise)

---

## 🔧 Technical Details

### Dependencies

```
Flask==2.3.3              # Web framework
pandas==2.0.3             # Data processing
scikit-learn==1.3.0       # Machine learning
numpy==1.24.3             # Numerical computing
PySastrawi==1.2.0         # Indonesian NLP
```

### API Endpoints

#### POST `/api/classify`
**Request**:
```json
{
  "tweet": "lu goblok ni bodoh banget"
}
```

**Response**:
```json
{
  "success": true,
  "prediction": "Umpatan Kasar & Binatang",
  "confidence": 0.92,
  "description": "Harsh words and animal insults directed at someone",
  "color": "#FF6B6B",
  "preprocessed_text": "goblok bodoh",
  "model": "SVM",
  "accuracy": 0.89
}
```

#### GET `/api/info`
Returns model metadata and performance metrics.

#### GET `/`
Serves the web interface (index.html).

---

## 🔄 Retraining

If you have new data:

```bash
# 1. Prepare your data (same format as Merged_Preprocessed_Cyberbullying.xlsx)

# 2. Run preprocessing
python preprocess.py

# 3. Train new models
python train_models.py

# 4. New model files will be saved:
#    - best_cyberbullying_model.pkl
#    - tfidf_vectorizer.pkl
#    - model_metadata.json
```

---

## 📈 Performance Monitoring

The app logs:
- Prediction time (typically < 100ms)
- Model accuracy on test set
- All 6 category predictions

Monitor in the terminal output when `python app.py` is running.

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'flask'` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: best_cyberbullying_model.pkl` | Verify model file is in same folder as app.py |
| `Address already in use` | Change port in app.py: `app.run(port=8080)` |
| `Sastrawi not working` | Run `pip install PySastrawi --upgrade` |
| Model predictions are slow | This is normal first request (~500ms), cache warms up |

---

## 📚 Project Structure Explained

### `app.py` - Main Application
- Loads trained model and vectorizer
- Defines Flask routes and API endpoints
- Implements preprocessing pipeline
- Handles HTTP requests from browser

### `train_models.py` - Training Pipeline
- Loads preprocessed data
- Performs train/test split
- Extracts TF-IDF features
- Trains 3 classifiers
- Evaluates and saves best model

### `preprocess.py` - Data Preprocessing
- Merges 3 raw data files
- Maps keywords to 6 categories
- Applies 5-step preprocessing pipeline
- Saves preprocessed datasets

### `templates/index.html` - Web Interface
- Beautiful responsive UI
- Handles user input
- Calls API endpoints via JavaScript
- Displays results with styling

### Model Files
- `best_cyberbullying_model.pkl` - Serialized SVM model
- `tfidf_vectorizer.pkl` - Fitted TF-IDF vectorizer
- `model_metadata.json` - Performance metrics and categories

### Data Files
- `Merged_Preprocessed_Cyberbullying.xlsx` - Full pipeline trace
- `ML_Ready_Cyberbullying.csv` - Training data

---

## 📝 For Your Pre-Thesis Report

This project demonstrates:
- **Data Science**: Preprocessing, feature extraction, ML pipeline
- **Machine Learning**: Model selection, evaluation, deployment
- **Web Development**: Flask backend, HTML/CSS/JavaScript frontend
- **NLP**: Indonesian text processing with Sastrawi

### Key Sections to Document
1. **Introduction**: Problem statement on cyberbullying detection
2. **Methodology**: 5-step preprocessing, TF-IDF, SVM classification
3. **Dataset**: 2,036 tweets, 6 categories, 80/20 split
4. **Results**: 89% accuracy, per-category performance
5. **Implementation**: Web application architecture
6. **Conclusion**: Summary and future improvements

---

## 👥 Team Notes

- **Data Engineer**: Focus on preprocessing and feature extraction
- **ML Engineer**: Model training and evaluation (all done!)
- **Frontend Developer**: Customize HTML/CSS styling
- **Backend Developer**: Extend app.py with database, logging
- **DevOps**: Deploy to cloud using Heroku/AWS

---

## 📞 Support

If you encounter issues:

1. Check SETUP_GUIDE.md for detailed instructions
2. Verify all files are in correct locations
3. Ensure Python dependencies are installed
4. Check that port 5000 is not in use
5. Review terminal output for error messages

---

## ✅ Checklist Before Presentation

- [ ] All files downloaded and organized
- [ ] `pip install -r requirements.txt` completed
- [ ] `python app.py` runs without errors
- [ ] Can access http://localhost:5000 in browser
- [ ] Can classify sample tweets successfully
- [ ] Results display with correct categories and confidence
- [ ] Preprocessing steps visible in UI
- [ ] Team understands the pipeline
- [ ] Ready for live demo

---

## 📜 License

Educational project for BINUS University Pre-Thesis Research
Department of Information Systems

---

## 🎓 References

- Sastrawi: https://github.com/har07/PySastrawi
- Scikit-learn: https://scikit-learn.org/
- Flask: https://flask.palletsprojects.com/
- TF-IDF: https://nlp.stanford.edu/IR-book/

---

**Last Updated**: May 12, 2026  
**Status**: ✅ Production Ready  
**Accuracy**: 89%  
**Model**: Support Vector Machine (SVM)

Good luck with your pre-thesis presentation! 🎉

