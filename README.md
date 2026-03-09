# MSIS 522 HW1 - Prompt Attack Detection App

This project presents an end-to-end machine learning workflow for detecting prompt attacks in LLM systems. The app was built in Streamlit and summarizes the full workflow across descriptive analytics, predictive modeling, SHAP explainability, and interactive prediction.

## Project Summary

The dataset comes from the Hugging Face `neuralchemy/Prompt-injection-dataset`, which was reframed into a 3-class classification problem:

- Benign
- Jailbreak
- Injection/Exfil/Tool Hijack

The final engineered dataset contains 6,274 rows and 18 interpretable numeric/binary features. Instead of embeddings, the feature set uses counts, ratios, and indicator-style variables such as:

- word count
- line count
- punctuation density
- caps ratio
- override phrase count
- secret keyword count
- exfiltration phrase count

## Why This Matters

Prompt attacks can override safeguards, manipulate tool use, and attempt to extract hidden instructions or sensitive information. Detecting them matters for enterprise AI systems connected to retrieval, tools, and private knowledge.

## Models Compared

The following models were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Multi-Layer Perceptron (MLP)

### Best Model

XGBoost was the best-performing overall model based on macro F1.

## Streamlit App Features

The app contains 4 tabs:

1. Executive Summary  
2. Descriptive Analytics  
3. Model Performance  
4. Explainability & Interactive Prediction  

The final tab allows a user to:
- choose a trained model
- set custom feature values
- generate a live prediction
- view class probabilities
- view a local SHAP waterfall explanation using XGBoost

## Files Included

This repository includes:

- `app.py` — main Streamlit application
- saved model artifacts (`.joblib`, `.keras`)
- processed data files used by the app
- saved best-parameter JSON files
- saved MLP training-history images
- `requirements.txt`

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py