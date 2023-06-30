# 🎣 PhishNet - Comprehensive Phishing Detection System 🚫

PhishNet is a comprehensive phishing detection system that harnesses the power of machine learning, natural language processing, and distributed computing to accurately identify and classify phishing attempts.

## 🌐 DARTH Framework
PhishNet operates based on the Distributed Analysis for Research and Threat Hunting (DARTH) framework. This system smartly analyses email content to detect potential phishing threats, showcasing an impressive 98% success rate.

![](https://github.com/jag-prabhakaran/PhishNet/assets/73809351/1fe6f88d-7cab-4310-ad33-e8e376ef5693)


## 🛠 Features

- **Semantic Analysis**: PhishNet uses a pre-trained BERT model for encoding email content into dense vector representations. These representations conserve the semantic details present in the text, providing an effective means of predicting the probability of an email being a phishing attempt.

- **URL-based Detection**: By vectorizing URLs into numerical representations, PhishNet employs K-Nearest Neighbor (KNN) modeling to effectively identify phishing URLs with an impressive precision rate of 92%.

- **Combined Predictions**: The system integrates predictions from email content analysis and URL-based detection using an Artificial Neural Network (ANN). This combined model boosts the overall accuracy of phishing detection.

## 📝 Files in this project
- `ann.py`: Implementation of the artificial neural network model.
- `knn.py`: Implementation of the K-Nearest Neighbor model for URL-based detection.
- `phishnetbert.ipynb`: Implementation of the BERT model for semantic analysis.
- `predicturl.py`: Module responsible for URL prediction.
- `preprocessing.py`: Module for data preprocessing.

## ⚡ Quickstart
1. Clone the project repository on your local machine.
2. Install the required dependencies using pip: `pip install -r requirements.txt`
3. Execute the desired Python script or Jupyter notebook.

Be safe, secure, and phishing-free with PhishNet! 🎣🚫
