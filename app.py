from flask import Flask, render_template, request
import pandas as pd
import joblib
from classes import Converter
from urllib.parse import urlparse
from nltk.tokenize import RegexpTokenizer
import tldextract
from sklearn.base import BaseEstimator, TransformerMixin
import requests
import re

app = Flask(__name__)



@app.route('/')
def index():
    return render_template("account-chat.html")



def parse_url(url: str):
    try:
        no_scheme = not url.startswith('https://') and not url.startswith('http://')
        if no_scheme:
            parsed_url = urlparse(f"http://{url}")
            return {
                "url": url,
                "scheme": None,
                "netloc": parsed_url.netloc,
                "path": parsed_url.path,
                "params": parsed_url.params,
                "query": parsed_url.query,
                "fragment": parsed_url.fragment,
            }
        else:
            parsed_url = urlparse(url)
            return {
                "url": url,
                "scheme": parsed_url.scheme,
                "netloc": parsed_url.netloc,
                "path": parsed_url.path,
                "params": parsed_url.params,
                "query": parsed_url.query,
                "fragment": parsed_url.fragment,
            }
    except:
        return None


def get_num_subdomains(netloc: str):
    subdomain = tldextract.extract(netloc).subdomain
    if subdomain == "":
        return 0
    return subdomain.count('.') + 1

def tokenize_domain(netloc: str):
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    split_domain = tldextract.extract(netloc)
    no_tld = str(split_domain.subdomain +'.'+ split_domain.domain)
    return " ".join(map(str,tokenizer.tokenize(no_tld)))

def load_model(filename):
    return joblib.load(filename)

import sys
sys.modules["__main__"].Converter = Converter
clf = load_model('naive_bayes_classifier.pkl')


def predict_url(model, url: str):
    parsed_url = parse_url(url)
    data = pd.DataFrame.from_records([parsed_url])
    data["length"] = data.url.str.len()
    data["tld"] = data.netloc.apply(lambda nl: tldextract.extract(nl).suffix)
    data['tld'] = data['tld'].replace('','None')
    data["is_ip"] = data.netloc.str.fullmatch(r"\d+\.\d+\.\d+\.\d+")
    data['domain_hyphens'] = data.netloc.str.count('-')
    data['domain_underscores'] = data.netloc.str.count('_')
    data['path_hyphens'] = data.path.str.count('-')
    data['path_underscores'] = data.path.str.count('_')
    data['slashes'] = data.path.str.count('/')
    data['full_stops'] = data.path.str.count('.')
    data['num_subdomains'] = data['netloc'].apply(lambda net: get_num_subdomains(net))
    data['domain_tokens'] = data['netloc'].apply(lambda net: tokenize_domain(net))
    data['path_tokens'] = data['path'].apply(lambda path: " ".join(map(str,tokenizer.tokenize(path))))
    data.drop(['url', 'scheme', 'netloc', 'path', 'params', 'query', 'fragment'], axis=1, inplace=True)

    pred = model.predict(data)
    pred_proba = model.predict_proba(data)

    return pred[0], max(pred_proba[0])


tokenizer = RegexpTokenizer(r'[A-Za-z]+')

preprocessor = joblib.load('preprocessor.pkl')

API_URL = "https://api-inference.huggingface.co/models/Jagannath/phishNet"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def extract_urls(text):
    # Regular expression pattern to find URLs starting with 'www.' or 'https://'
    url_pattern = r'\b(https?://[^\s]+|www\.[^\s]+)'

    # Find all URLs in the text using the pattern
    urls = re.findall(url_pattern, text)

    return urls

def get_predictions(url, text):
    url_prediction, url_confidence = predict_url(clf, url)

    url_list = extract_urls(text)
    goodCount = 0
    badCount = 0
    for url in url_list:
      prediction, confidence = predict_url(clf, url)
      if prediction == 'good':
        goodCount += 1
      else:
        badCount += 1
    url_pred = None
    if goodCount > badCount:
      url_pred = 'good'
    elif badCount > goodCount:
      url_pred = 'bad'
    elif goodCount == badCount == 0:
      url_pred = None

    if url_pred != None:
      url_prediction = url_pred

    # Text prediction
    output = query({
    "inputs": text,
    "parameters": {
      "truncation": "only_first"
    }
    })

    max_confidence_label = max(output[0], key=lambda x: x['score'])
    predicted_label = "good" if max_confidence_label['label'] == 'LABEL_0' else "bad"
    text_confidence_score = max_confidence_label['score']

    return url_prediction, url_confidence, predicted_label, text_confidence_score

def ensemble_prediction(url, text):
    url_prediction, url_confidence, text_prediction, text_confidence = get_predictions(url, text)
    if url_prediction == "None":
      print(url_prediction + " : " + str(url_confidence))
    if text_prediction == "None":
      print(text_prediction + " : " + str(text_confidence))
    if url_prediction == text_prediction:
        print(url_prediction, (url_confidence + text_confidence)/2)
        return url_prediction, (url_confidence + text_confidence)/2
    else:
        if url_confidence > text_confidence:
            print(url_prediction, (url_confidence + text_confidence)/2)
            return url_prediction, (url_confidence + text_confidence)/2
        else:
            print(text_prediction, (url_confidence + text_confidence)/2)
            return text_prediction, (url_confidence + text_confidence)/2
        
        
        
@app.route('/form', methods=["POST"])
def form():
    email_body = request.form.get("email_body")
    embedded_url = request.form.get("url")
    prediction, confidence = ensemble_prediction(embedded_url, email_body)
    return render_template("form.html", email_body=email_body, embedded_url=embedded_url, prediction=prediction, confidence=int(confidence * 100))

if __name__ == "__main__":
    app.run(debug=True)