from urllib.parse import urlparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
import joblib



loaded_model = joblib.load('knn_model.pkl')

def parse_and_vectorize_url(url):
    parsed_url = urlparse(url)
    url_tokens = parsed_url.netloc.split('.')
    url_text = ' '.join(url_tokens)
    url_vector = cv.transform([url_text])
    return url_vector

url = 'paypal.com.cgi.bin.webscr.cmd.login.submit.dispatch.5885d80a13c03faee8dcbcd55a50598f04d34b4bf5tt1.mediareso.com/secure-code90/security/'

cv = CountVectorizer()
cv.fit(df['text_sent'])

parsed_vector = parse_and_vectorize_url(url)

model = KNeighborsClassifier(n_neighbors=4)
model.fit(Xtrain, Ytrain)

prediction = model.predict(parsed_vector)
print(prediction)