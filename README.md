# 📊 TF-IDF Based Sentiment Analysis using Classical Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Classical-green)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-orange)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

---

## 📌 Project Overview

This project performs **Sentiment Analysis** on Twitter data using:

- **TF-IDF (Term Frequency – Inverse Document Frequency)**
- Classical Machine Learning algorithms

The model classifies tweets into:

- ✅ Positive  
- ❌ Negative  

This project demonstrates a complete NLP pipeline from preprocessing to evaluation.

---

## 🎯 Objective

- Convert raw text into numerical vectors using TF-IDF
- Train classical ML models
- Evaluate performance using standard metrics
- Build a simple and interpretable NLP model

---

## 📂 Project Structure

```
TF-IDF-Based-Sentiment-Analysis/
│
├── tf-idf-based-sentiment-analysis-with-classical-ml.ipynb
├── README.md
```

---

## 📊 Dataset

This project uses the **Sentiment140 Dataset** from Kaggle.

🔗 Kaggle Dataset Link:  
https://www.kaggle.com/datasets/kazanova/sentiment140

The dataset contains **1.6 million labeled tweets** for sentiment classification.

---


### 💻 Local System Setup

If running locally:

1. Download dataset from Kaggle.
2. Extract the CSV file.
3. Place it inside your project folder.
4. Use:

---

## ⚙️ Workflow

### 1️⃣ Data Loading
- Load dataset using Pandas
- Inspect shape and preview records

### 2️⃣ Data Preprocessing
- Convert text to lowercase
- Remove punctuation
- Remove stopwords
- Clean special characters

### 3️⃣ Feature Extraction (TF-IDF)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(text_data)
```

### 4️⃣ Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 5️⃣ Model Training

```python
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)
```

### 6️⃣ Model Evaluation

```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## 📈 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

---

## 🛠️ Technologies Used

- Python 3.x  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Jupyter Notebook  

---

## 🚀 How to Run

1️⃣ Clone the repository:

```
git clone https://github.com/your-username/your-repo-name.git
```

2️⃣ Install dependencies:

```
pip install -r requirements.txt
```

3️⃣ Open Jupyter Notebook:

```
jupyter notebook
```

4️⃣ Run all cells step by step.

---

## 🔮 Future Improvements

- Add Deep Learning models (LSTM / BERT)
- Hyperparameter tuning
- Streamlit deployment
- Real-time sentiment prediction UI
- Model comparison (Naive Bayes vs Logistic Regression vs SVM)

---

## 🎓 Academic Use

Suitable for:

- Final Year AI/ML Project
- NLP Practical Implementation
- Machine Learning Portfolio
- Resume Project Showcase

---

## 👨‍💻 Author

**Ayush Kumar**  
Diploma in Artificial Intelligence & Machine Learning  
