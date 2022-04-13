import pandas as pd
import re
import warnings
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import random
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn import svm
warnings.simplefilter('ignore')
import pickle


def transform_dataset(serbian_data, english_data):
    # Loading the dataset
    # serbian_data = pd.read_csv('../data/serbian-reviews/serbian-reviews.csv', delimiter='|')
    # english_data = pd.read_csv('../data/english-reviews/english-reviews.csv', delimiter='|')

    data = []
    for review in serbian_data['review']:
        data.append((review, 0))

    for review in english_data['review']:
        data.append((review, 1))

    random.shuffle(data)
    return pd.DataFrame(data, columns=['Text', 'Language'])


def tfidf_initialization(X):
    tfidf_vectorizer = TfidfVectorizer(use_idf=False, lowercase=False, sublinear_tf=True, ngram_range=(1, 2))
    tfidf_vectorizer.fit(X)
    return tfidf_vectorizer


def train(serbian_data, english_data):
    df = transform_dataset(serbian_data, english_data)

    X = df['Text']
    y = df['Language']

    # creating a list for appending the preprocessed text
    data_list = []
    # iterating through all the text

    for text in X:
        # removing the symbols and numbers
        text = re.sub(r"[!@#$(),n'%^*?:;~`0-9]", ' ', text)
        text = re.sub(r'[[]]', ' ', text)
        # converting the text to lower case
        text = text.lower()
        # appending to data_list
        data_list.append(text)

    # train test splitting
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    tfidf_vectorizer = tfidf_initialization(X)
    x_train_tfidf = tfidf_vectorizer.transform(x_train)
    x_test_tfidf = tfidf_vectorizer.transform(x_test)

    # model creation and prediction
    model = MultinomialNB()
    # model = svm.SVC()
    model.fit(x_train_tfidf, y_train)
    # prediction
    predictions = model.predict(x_test_tfidf)
    # model evaluation
    score = accuracy_score(y_test, predictions)
    print(classification_report(y_test, predictions))

    return model, tfidf_vectorizer


# function for predicting language
def predict(text, vectorizer, model):
    x = vectorizer.transform([text])
    lang = model.predict(x)
    if lang[0] == 0:
        print(text, '>> Serbian')
    else:
        print(text, '>> English')
    return lang[0]


def train_and_test():
    # Loading the dataset
    serbian_data = pd.read_csv('../data/serbian-reviews/serbian-reviews.csv', delimiter='|')
    english_data = pd.read_csv('../data/english-reviews/english-reviews.csv', delimiter='|')

    df = transform_dataset(serbian_data, english_data)

    X = df['Text']
    y = df['Language']

    # creating a list for appending the preprocessed text
    data_list = []
    # iterating through all the text

    for text in X:
        # removing the symbols and numbers
        text = re.sub(r"[!@#$(),n'%^*?:;~`0-9]", ' ', text)
        text = re.sub(r'[[]]', ' ', text)
        # converting the text to lower case
        text = text.lower()
        # appending to data_list
        data_list.append(text)

    # train test splitting
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    tfidf_vectorizer = tfidf_initialization(X)
    x_train_tfidf = tfidf_vectorizer.transform(x_train)
    print(x_train_tfidf)
    x_test_tfidf = tfidf_vectorizer.transform(x_test)

    # model creation and prediction
    model = MultinomialNB()
    # model = svm.SVC()
    model.fit(x_train_tfidf, y_train)
    # prediction
    predictions = model.predict(x_test_tfidf)
    # model evaluation
    score = accuracy_score(y_test, predictions)
    print('Training results')
    print(classification_report(y_test, predictions))

    # with open('language_detection.pkl', 'wb') as sm:
    #     pickle.dump(model, sm)

    # testing
    serbian_data = pd.read_csv('../data/serbian-reviews/serbian-reviews-test.csv', delimiter='|')
    english_data = pd.read_csv('../data/english-reviews/english-reviews-test.csv', delimiter='|')

    # print(serbian_data)

    df = transform_dataset(serbian_data, english_data)

    X = df['Text']
    y = df['Language']

    # with open('language_detection.pkl', 'rb') as sm:
    #     model = pickle.load(sm)

    test_tfidf = tfidf_vectorizer.transform(X)

    # prediction
    predictions = model.predict(test_tfidf)
    # model evaluation
    score = accuracy_score(y, predictions)
    print('Testing results')
    print(classification_report(y, predictions))


if __name__ == '__main__':
    train_and_test()






