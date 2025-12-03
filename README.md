# SERIxShravani
This is my prototype for FIN 545 Risk Management for Financial Cybersecurity Course, here at Stevens

# SERI-x-Shravani  
### Scam Exposure Risk Index (SERI) – A Behavioral AI Prototype for Financial Cybersecurity  
**Author:** Shravani Sawant  
**Course:** FIN 545 – Risk Management for Financial Cybersecurity, Stevens Institute of Technology  

---

## Project Overview  
This repository contains my end-to-end prototype for detecting financial scams using a hybrid **Behavioral Lexicon Score (SERI)** and a **Machine Learning (TF-IDF + Logistic Regression)** model.  
The system classifies input text into:

- **Safe**
- **Suspicious**
- **High-Risk Fraud**

The prototype is implemented in Python and includes a fully working **Streamlit app**, pre-trained models, reproducible notebooks, and dataset design files.

---

## Features  
### 1. Scam Exposure Risk Index (SERI)  
A handcrafted behavioral scoring system based on:

- urgency patterns  
- psychological manipulation cues  
- financial intent markers  
- account takeover signals  
- payoff-based scam indicators  

SERI returns a risk score between **0 and 100**.

---

### 2. ML Classifier (TF-IDF + Logistic Regression)  
Model trained on multiple curated datasets:

- Phishing & social-engineering conversations  
- Financial fraud emails  
- Real-world phishing datasets  
- Synthetic FinTech-themed scam templates  

Model Output: **probability of scam (0–1)**.

---

### 3. Hybrid Final Score  
The app combines:

**SERI Score + ML Probability → Final Risk Category**

This hybrid approach avoids full reliance on ML and mirrors **real-world risk engines** used in banking systems.

---

##  Repository Structure  
FIN_Project/
│
├── app.py # Streamlit application
├── FIN545_Model.ipynb # Model training notebook
├── FIN545_build_dataset.ipynb # Dataset creation & preprocessing notebook
│
├── phishing_classifier.joblib # Trained ML model
├── tfidf_vectorizer.joblib # TF-IDF vectorizer
├── seri_weights.json # Weights used for SERI scoring
├── seri_lexicons.json # Keyword patterns for SERI
│
├── fintech_scam_500.csv # Clean consolidated dataset (small)
└── fintech_scam_full.csv # Full dataset used in experimentation


---

## Datasets Used  
All datasets were reviewed and curated from public, academic, and open-source sources.

### **Hugging Face Sources (Cited in my research paper)**  
- https://huggingface.co/datasets/amitkedia/Financial-Fraud-Dataset  
- https://huggingface.co/datasets/Ngadou/social-engineering-convo  
- https://huggingface.co/datasets/zefang-liu/phishing-email-dataset  
- https://huggingface.co/datasets/UniqueData/spam-text-messages-dataset  
- https://huggingface.co/AcuteShrewdSecurity/Llama-Phishsense-1B  

These sources were **not used directly**, but were **studied for patterns and dataset design**, as described in Section IV-A of the research paper.

---

## Installation & Setup

### **1. Clone the repository**
git clone https://github.com/shravanips/SERIxShravani.git
cd SERIxShravani

### **2. Create and activate environment**
python3 -m venv venv
source venv/bin/activate   # MacOS/Linux
venv\Scripts\activate      # Windows

### **3. Install dependencies**
pip install -r requirements.txt

If you did not upload requirements.txt, generate it using:
pip freeze > requirements.txt

### **Running the Streamlit App**
Once dependencies are installed, run:
streamlit run app.py

This will launch a local web app where users can:
- paste text/email/conversation
-view SERI behavioral breakdown
- view ML probability
- get final scam risk category

---

### **Model Architecture**
- TF-IDF Vectorizer (unigrams + bigrams)
- Logistic Regression with class weighting
- Custom SERI Lexicon Engine
- Hybrid Scoring Layer

This architecture balances interpretability (important in finance) and accuracy.

---

### **Ethical Note**
- This prototype is for academic and research purposes only.
- It is not intended for production fraud detection and does not store or process user data.
- All datasets used were sourced from public or open-source locations.

---

### **Acknowledgements**
This work was completed as part of:
FIN 545 — Risk Management for Financial Cybersecurity
Instructor: Prof. Paul Rohmeyer

---

## Contact
For questions or collaboration:

Shravani Sawant
- GitHub: https://github.com/shravanips
- LinkedIn: https://www.linkedin.com/in/shravanisawant03/
