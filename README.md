# Emotion Detection using ML, Deep Learning & Transformers

An end-to-end **NLP emotion classification project** comparing Traditional Machine Learning, Deep Learning, and Transformer-based approaches.

The project explores different text representations and model architectures to understand their performance on emotion detection.

---

##  Key Results

| Approach | Best Model | Test Accuracy |
|---|---|---:|
| Traditional ML | Logistic Regression / SVM | **89%** |
| Deep Learning | GRU / Stacked GRU | **92%** |
| Transformers | BERT-based experiment | Explored |

>  **Best completed result: 92% test accuracy with GRU-based models.**

---

##  Approach

```text
Raw Text
   ↓
Text Preprocessing
   ↓
Feature Engineering / Tokenization
   ↓
Traditional ML ──────┐
                     │
Deep Learning ───────┼──→ Model Evaluation → Comparison
                     │
Transformers ────────┘
```

### Traditional ML
- Bag of Words
- TF-IDF
- Word2Vec
- Logistic Regression
- SVM
- XGBoost
- Hyperparameter tuning

### Deep Learning
- RNN
- LSTM
- GRU
- Bidirectional RNN
- Bidirectional LSTM
- Bidirectional GRU
- Stacked architectures
- Hyperparameter tuning

### Transformers
- BERT-based experimentation
- Transformer tokenization and classification workflow

---

##  Model Comparison

### Traditional ML

| Model | Representation | Test Accuracy |
|---|---|---:|
| Logistic Regression | BOW | **89%** |
| SVM | BOW | 88% |
| Logistic Regression | TF-IDF | 87% |
| SVM | TF-IDF | **89%** |
| Tuned XGBoost | TF-IDF | 88% |

### Deep Learning

| Model | Test Accuracy |
|---|---:|
| LSTM | 90% |
| GRU | **92%** |
| Bi-LSTM | 90% |
| Bi-GRU | 91% |
| Stacked GRU | **92%** |
| Tuned Stacked GRU | **92%** |

Detailed experiment results are available in the notebooks and `results/` directory.

---

##  Key Findings

- Traditional ML models achieved competitive results using BoW and TF-IDF.
- Word2Vec experiments performed weaker than BoW/TF-IDF configurations.
- GRU-based architectures achieved the strongest completed results.
- **92% test accuracy** was achieved by GRU, Stacked GRU, and Tuned Stacked GRU.
- More complex architectures did not always improve generalization.
- The project demonstrates the progression from classical NLP to deep learning and transformers.

---

##  Tech Stack

**Language:** Python

**Machine Learning:** Scikit-learn, XGBoost

**Deep Learning:** TensorFlow, Keras

**NLP:** NLTK, BoW, TF-IDF, Word2Vec

**Transformers:** Hugging Face Transformers, BERT

**Development:** Jupyter Notebook, Git, GitHub


##  Evaluation

Models were primarily compared using **test accuracy**.

Training and validation performance were also reviewed to understand model generalization and potential overfitting.

---

##  Future Improvements

- Complete transformer fine-tuning and evaluation
- Add precision, recall, F1-score, and confusion matrices
- Build a Streamlit inference application
- Create a reusable prediction pipeline
- Deploy the best-performing model

---

##  Author

**Archit Tomar**

AI/ML | NLP | Deep Learning | Transformers

---

 If you find this project useful, consider giving the repository a star.
