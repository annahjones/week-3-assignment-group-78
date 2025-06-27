# week-3-assignment-group-78
# 🧠 AI Assignment: Joins, Models & Real-World Implementation

## 📋 Overview

This project covers core concepts in AI and Machine Learning using real-world datasets. It includes theoretical analysis, practical implementation with Python and SQL, and ethical considerations in model development and deployment.

---

## 📌 Contents

### ✅ Part 1: Theoretical Understanding

- Differences between TensorFlow and PyTorch
- Use cases for Jupyter Notebooks
- spaCy's role in enhancing NLP tasks
- Comparison of Scikit-learn and TensorFlow

### 💻 Part 2: Practical Implementation

#### 1. Classical ML with Scikit-learn (Iris Dataset)
- Preprocessing
- Training a Decision Tree classifier
- Evaluation (Accuracy, Precision, Recall)

#### 2. Deep Learning with TensorFlow (MNIST Dataset)
- Building and training a CNN
- Test accuracy > 95%
- Visualization of predictions on sample images

#### 3. NLP with spaCy (Amazon Reviews)
- Named Entity Recognition (NER)
- Rule-based Sentiment Analysis

### 🔍 Part 3: Ethics & Optimization

- Identifying potential bias in models
- Tools for fairness (TensorFlow Fairness Indicators, spaCy rule-based systems)
- Debugging TensorFlow code

### 🌐 Bonus Task: Web Deployment

- Deployment of MNIST classifier using **Streamlit**
- Screenshot and live demo link provided

---

## 🚀 Requirements

- Python 3.x  
- Jupyter Notebook or any Python IDE  
- Libraries:
  - `scikit-learn`
  - `tensorflow` / `keras`
  - `spacy`
  - `matplotlib`
  - `streamlit` (for deployment)
- MySQL or any SQL environment

Install dependencies:
```bash
pip install scikit-learn tensorflow spacy matplotlib streamlit
python -m spacy download en_core_web_sm

## File structure
AI_Assignment/
│
├── part1_theory_answers.md
├── iris_decision_tree.ipynb
├── mnist_cnn.py
├── amazon_reviews_nlp.py
├── README.md
├── answers.sql
├── mnist_web_app/  # Streamlit app files
│   └── app.py
└── screenshots/
    └── streamlit_demo.png
