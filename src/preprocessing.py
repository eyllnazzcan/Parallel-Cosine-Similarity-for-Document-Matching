import re
import nltk
import string
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
tokenizer = TreebankWordTokenizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+|www\\S+", "", text)
    text = re.sub(r"[^a-z\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    
    tokens = tokenizer.tokenize(text)
    cleaned = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words and len(token) > 1
    ]
    return " ".join(cleaned)

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
docs = [clean_text(doc) for doc in newsgroups.data if doc.strip() != '']

with open("AllCombined_cleaned.txt", "w", encoding="utf-8") as f:
    for doc in docs:
        f.write(doc + "\n")
