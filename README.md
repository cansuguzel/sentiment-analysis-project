# Sentiment Analysis of Movie Reviews

This repository contains two approaches for **sentiment analysis** on movie reviews:

1. **Classical Approach:** TF-IDF vectorization + Logistic Regression
2. **Modern Approach:** BERT (transformer-based model) for deep contextual understanding

The project aims to classify movie reviews as **positive** or **negative**.

## Project Overview

* **Logistic Regression:**

  * Text preprocessing: tokenization, stopword removal, TF-IDF vectorization
  * Simple and interpretable baseline classifier
  * Performs reasonably well but struggles with contextual nuances such as negation

* **BERT:**

  * Pretrained BERT tokenizer for tokenization
  * Fine-tuned BERT for sequence classification
  * Handles context and nuanced language (e.g., negations, sarcasm) effectively

---

## Dataset

We use the **[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)**.

* Balanced dataset with 25,000 positive and 25,000 negative reviews
* Train/Test split: 80/20

---

## Installation

### Logistic Regression

```bash
git clone https://github.com/cansuguzel/sentiment-analysis-project.git
cd sentiment-analysis-project
pip install -r requirements_logistic.txt
```

### BERT

```bash
pip install -r requirements_bert.txt
```
## Usage

### Logistic Regression

Open and run the notebook:  
[Sentiment_Analyze_with_LogisticRegression.ipynb](https://github.com/cansuguzel/sentiment-analysis-project/blob/main/Sentiment_Analyze_with_LogisticRegression.ipynb)

* Trains the classical model
* Evaluates performance on test data

### BERT

Open and run the notebook:  
[Sentiment_Analyze_with_BERT.ipynb](https://github.com/cansuguzel/sentiment-analysis-project/blob/main/Sentiment_Analyze_with_BERT_clean.ipynb)

* Fine-tunes BERT on the sentiment dataset
* Evaluates performance on test data


---

## Results

### Logistic Regression Results

**Classification Report**

```
Accuracy: 0.8832
              precision    recall  f1-score   support
    negative       0.90      0.87      0.88      4961
    positive       0.87      0.90      0.89      5039
    accuracy                           0.88     10000
   macro avg       0.88      0.88      0.88     10000
weighted avg       0.88      0.88      0.88     10000
```

**Confusion Matrix**

```
[[4294  667]
 [ 501 4538]]
```

**Example Prediction**

* Input: `"The movie wasn’t so bad, actually I enjoyed it a lot"`
* Predicted: **negative** (fails to capture contextual negation)

---

### BERT Results

**Classification Report**

```
Accuracy: 0.898
              precision    recall  f1-score   support
    negative       0.90      0.87      0.89      4961
    positive       0.88      0.90      0.89      5039
    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000
```

**Example Prediction**

* Input: `"The movie wasn’t so bad, actually I enjoyed it a lot"`
* Predicted: **positive** (captures negation and context correctly)

---

### Comparison

| Model               | Accuracy | Strengths                         | Weaknesses                |
| ------------------- | -------- | --------------------------------- | ------------------------- |
| Logistic Regression | 0.8832   | Simple, interpretable, fast       | Struggles with context    |
| BERT                | 0.898    | Handles context, negation, nuance | Slower, heavier resources |

---

## Project Insights

* Logistic Regression provides a solid **baseline**. Good for quick experiments.
* BERT improves performance for **context-dependent** sentiment, making it suitable for nuanced NLP tasks.
* Fine-tuning pretrained transformers is essential for **capturing subtle language patterns**.



## Contact

* GitHub: [cansuguzel](https://github.com/cansuguzel)
