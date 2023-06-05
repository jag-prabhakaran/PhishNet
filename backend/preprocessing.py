import re
import nltk
from nltk.corpus import stopwords
import spacy
import textacy
import contractions

punctuations = '''~!@#$%^&*()"_+-=`;':,./<>?'''

text = '''
Dear PayPal User,

We regret to inform you that your PayPal account has been compromised and we have noticed some unauthorized activities. To protect your account, we kindly request you to verify your information immediately.

Click on the following link to verify your account: paypalredirect.co
b
Please note that failure to verify your account within 24 hours will result in permanent account suspension.

Thank you for your cooperation.

Best regards,
PayPal Security Team 
'''

#Contraction fixing
text = contractions.fix(text)

#removing punctuation + lowercase
text = re.sub(r'[^\w\s]', '',text).lower()

nltk.download('punkt')
nltk.download('wordnet')

#tokenization
tokenized_text = nltk.word_tokenize(text)

print(tokenized_text)

nltk.download('stopwords')
stop_words = stopwords.words('english')

#Removing Stop words
filtered_tokens = []
for token in tokenized_text:
    if token.lower() not in stop_words:
        filtered_tokens.append(token)
        
print(filtered_tokens)

#lemmatization
nlp = spacy.load("en_core_web_sm")

lem_words = []
for word in filtered_tokens:
    doc = nlp(word)
    for token in doc:
        lem_words.append(token.lemma_)
    
print(lem_words)
