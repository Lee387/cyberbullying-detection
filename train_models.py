#!/usr/bin/env python3
"""
Training Pipeline: TF-IDF + ML Classification
Trains Naive Bayes, SVM, and Random Forest classifiers
Evaluates all three and saves the best model + TF-IDF vectorizer
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import json

print("=" * 70)
print("FEATURE EXTRACTION & MODEL TRAINING PIPELINE")
print("=" * 70)

# ─── LOAD PREPROCESSED DATA ─────────────────────────────────────────────────────
print("\n[1/5] Loading preprocessed data...")
df = pd.read_csv('/mnt/user-data/outputs/ML_Ready_Cyberbullying.csv')

# Remove NaN values
df = df.dropna(subset=['Text_Preprocessed', 'Kategori_Cyberbullying'])

X = df['Text_Preprocessed'].values  # Input: preprocessed text
y = df['Kategori_Cyberbullying'].values  # Target: cyberbullying category

print(f"  ✓ Loaded {len(df)} preprocessed tweets")
print(f"  ✓ Features shape: {X.shape}")
print(f"  ✓ Target classes: {np.unique(y)}")
print(f"\n  Class distribution:")
for cat, count in pd.Series(y).value_counts().items():
    pct = count / len(y) * 100
    print(f"    • {cat}: {count} ({pct:.1f}%)")

# ─── TRAIN/TEST SPLIT ───────────────────────────────────────────────────────────
print("\n[2/5] Splitting data (80% train / 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  ✓ Training set: {len(X_train)} samples")
print(f"  ✓ Testing set: {len(X_test)} samples")

# ─── FEATURE EXTRACTION: TF-IDF ──────────────────────────────────────────────────
print("\n[3/5] TF-IDF Feature Extraction...")
tfidf = TfidfVectorizer(
    max_features=5000,      # Keep top 5000 features
    min_df=2,               # Min doc frequency
    max_df=0.95,            # Max doc frequency (remove very common terms)
    ngram_range=(1, 2),     # Unigrams + bigrams
    lowercase=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"  ✓ TF-IDF vectorizer fitted on training data")
print(f"  ✓ Feature matrix shape: {X_train_tfidf.shape}")
print(f"  ✓ Vocabulary size: {len(tfidf.get_feature_names_out())}")

# ─── TRAIN THREE CLASSIFIERS ────────────────────────────────────────────────────
print("\n[4/5] Training classifiers...")
print("  (This may take a minute...)\n")

models = {}
results = {}

# Naive Bayes
print("  Training Naive Bayes...")
nb = MultinomialNB(alpha=1.0)
nb.fit(X_train_tfidf, y_train)
y_pred_nb = nb.predict(X_test_tfidf)
acc_nb = accuracy_score(y_test, y_pred_nb)
prec_nb = precision_score(y_test, y_pred_nb, average='weighted', zero_division=0)
rec_nb = recall_score(y_test, y_pred_nb, average='weighted', zero_division=0)
f1_nb = f1_score(y_test, y_pred_nb, average='weighted', zero_division=0)
models['Naive Bayes'] = nb
results['Naive Bayes'] = {'accuracy': acc_nb, 'precision': prec_nb, 'recall': rec_nb, 'f1': f1_nb}
print(f"    ✓ Accuracy: {acc_nb:.4f} | Precision: {prec_nb:.4f} | Recall: {rec_nb:.4f} | F1: {f1_nb:.4f}")

# SVM
print("  Training SVM (Support Vector Machine)...")
svm = LinearSVC(max_iter=2000, random_state=42, dual=False)
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_test_tfidf)
acc_svm = accuracy_score(y_test, y_pred_svm)
prec_svm = precision_score(y_test, y_pred_svm, average='weighted', zero_division=0)
rec_svm = recall_score(y_test, y_pred_svm, average='weighted', zero_division=0)
f1_svm = f1_score(y_test, y_pred_svm, average='weighted', zero_division=0)
models['SVM'] = svm
results['SVM'] = {'accuracy': acc_svm, 'precision': prec_svm, 'recall': rec_svm, 'f1': f1_svm}
print(f"    ✓ Accuracy: {acc_svm:.4f} | Precision: {prec_svm:.4f} | Recall: {rec_svm:.4f} | F1: {f1_svm:.4f}")

# Random Forest
print("  Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train_tfidf.toarray(), y_train)  # RF needs dense matrix
y_pred_rf = rf.predict(X_test_tfidf.toarray())
acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, average='weighted', zero_division=0)
rec_rf = recall_score(y_test, y_pred_rf, average='weighted', zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)
models['Random Forest'] = rf
results['Random Forest'] = {'accuracy': acc_rf, 'precision': prec_rf, 'recall': rec_rf, 'f1': f1_rf}
print(f"    ✓ Accuracy: {acc_rf:.4f} | Precision: {prec_rf:.4f} | Recall: {rec_rf:.4f} | F1: {f1_rf:.4f}")

# ─── MODEL EVALUATION & SELECTION ───────────────────────────────────────────────
print("\n[5/5] Model Evaluation & Selection")
print("\n" + "=" * 70)
print("COMPARISON TABLE")
print("=" * 70)

comparison_df = pd.DataFrame(results).T
print(comparison_df.to_string())

best_model_name = comparison_df['f1'].idxmax()
best_model = models[best_model_name]
print(f"\n✓ BEST MODEL: {best_model_name} (F1-Score: {comparison_df.loc[best_model_name, 'f1']:.4f})")

# ─── DETAILED CLASSIFICATION REPORT ─────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"DETAILED REPORT: {best_model_name}")
print("=" * 70)

if best_model_name == 'Naive Bayes':
    y_pred = y_pred_nb
elif best_model_name == 'SVM':
    y_pred = y_pred_svm
else:
    y_pred = y_pred_rf

print("\nPer-Class Metrics:")
print(classification_report(y_test, y_pred, digits=4))

# ─── SAVE MODELS FOR DEPLOYMENT ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SAVING MODELS FOR DEPLOYMENT")
print("=" * 70)

# Save the best model
model_path = '/mnt/user-data/outputs/best_cyberbullying_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"  ✓ Best model saved: {model_path}")

# Save the TF-IDF vectorizer
tfidf_path = '/mnt/user-data/outputs/tfidf_vectorizer.pkl'
with open(tfidf_path, 'wb') as f:
    pickle.dump(tfidf, f)
print(f"  ✓ TF-IDF vectorizer saved: {tfidf_path}")

# Save metadata (category labels + model info)
metadata = {
    'model_name': best_model_name,
    'categories': list(np.unique(y)),
    'feature_extraction': 'TF-IDF (5000 features, ngram_range=(1,2))',
    'metrics': {
        'accuracy': results[best_model_name]['accuracy'],
        'precision': results[best_model_name]['precision'],
        'recall': results[best_model_name]['recall'],
        'f1_score': results[best_model_name]['f1']
    },
    'all_models_comparison': {
        'Naive Bayes': {k: float(v) for k, v in results['Naive Bayes'].items()},
        'SVM': {k: float(v) for k, v in results['SVM'].items()},
        'Random Forest': {k: float(v) for k, v in results['Random Forest'].items()}
    }
}

metadata_path = '/mnt/user-data/outputs/model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  ✓ Metadata saved: {metadata_path}")

# ─── SUMMARY ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DEPLOYMENT READY!")
print("=" * 70)
print(f"""
The following files are ready for your web application:

1. best_cyberbullying_model.pkl
   - Trained {best_model_name} classifier
   - Use: model = pickle.load(open('best_cyberbullying_model.pkl', 'rb'))

2. tfidf_vectorizer.pkl
   - Fitted TF-IDF vectorizer
   - Use: vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

3. model_metadata.json
   - Model info, categories, performance metrics

Web Application Flow:
  1. User pastes tweet → System applies same 5 preprocessing steps
  2. Text vectorized using saved TF-IDF
  3. {best_model_name} model predicts category
  4. Display result with confidence scores

Test Accuracy on holdout set: {results[best_model_name]['accuracy']:.1%}
""")
