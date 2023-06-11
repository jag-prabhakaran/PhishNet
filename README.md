# PhishNet

PhishNet is a comprehensive phishing detection system that combines multiple techniques and models to accurately identify and classify phishing attempts. The project leverages machine learning, natural language processing, and distributed computing to achieve high accuracy in phishing detection.

## DARTH Framework:
![57DADAA9-9AE8-4966-8BB9-33FA6D7ACBF3_1_201_a](https://github.com/jag-prabhakaran/PhishNet/assets/73809351/b8c1bea1-056d-4555-ac8a-e38974d403cb)



## Features

- DARTH framework: Devised the Distributed Analysis of Email Content (DARTH) framework, which analyzes the content of emails to detect phishing attempts. The framework achieves an impressive accuracy rate of 98% in identifying phishing emails.

- Semantic characteristics: PhishNet employs a pre-trained BERT model to encode the email content into dense vector representations. These representations preserve the semantic information present in the text, enabling PhishNet to effectively model the likelihood of an email being a phishing attempt. The pre-trained BERT model has been fine-tuned using a labeled dataset containing examples of legitimate and phishing emails. By training BERT on this dataset, PhishNet enhances its ability to extract meaningful features from email content and make accurate predictions.

- URL-based detection: PhishNet incorporates vectorization techniques to analyze and model URLs for phishing detection. By transforming URLs into numerical representations, the system gains the ability to leverage machine learning algorithms for classification. Integrated K-Nearest Neighbor (KNN) modeling for URL-based phishing detection. This approach achieves a precision rate of 92% in identifying phishing URLs.

- Combined model predictions: PhishNet combines predictions from the email content analysis and URL-based detection using an Artificial Neural Network (ANN). This combined model enhances the overall accuracy of phishing detection.
