import pandas as pd
from spellchecker import SpellChecker
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize,pos_tag
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

emoji_dict = {
    'excellent': [':-)', ':)', ':-d', ':d', 'xd', '😀', '😁', '😃', '😄', '😆', '😉', '😊', '😋', '😎', '😍', '😘', '🥰', '😗', '😙', '😚', '☺', '🙂', '🤗', '🤩', '😌', '😛', '🙃', '😏', '🤤', '🙄', '5+', '10+', '5.00', '5/5', '10/10', '👍', '👌'],
    'great': ['<3', '❤', '🧡', '💛', '💚', '💙', '💜', '🤎', '🖤', '🤍', '❣', '💕', '💞', '💓', '💗', '💖', '💘', '💝', '💟', '💌'],
    'bad': [':(', ':-(', '</3', '💔', '🤔', '🤨', '😐', '😑', '😶', '😥', '🤐', '😫', '🥱', '😴', '😒', '😓', '😔', '😕', '🤑', '☹', '🙁', '😖', '😞', '😟', '😢', '😭', '😦', '😧', '😨', '😩', '😬', '😰', '🥵', '😱', '😡', '🥺', '😣', '😠', '🤬', '🤮', '🤒', '🤢', '🤧', '🥶', '😤'],
    'ok': ['😂', '🤣'],
    '': [';)', ';-)', 'D:']
}

def load(path):
    return pd.read_csv(path, delimiter='|', encoding='utf-8', names=['title', 'review', 'rating', 'sentiment'], skiprows=1)

def remove_whitespace(text):
    return  " ".join(text.split())

def replace_special_chars(review):
    for c in '.,!?-:();*="\'\\/#':
        review = review.replace(c, ' ')
    return review

def replace_emoji(review):
    for sentiment, emoji_list in emoji_dict.items():
        for emoji in emoji_list:
            review = review.replace(emoji, ' ' + sentiment + ' ')
    return review

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def lemmatization(text):
    wordnet = WordNetLemmatizer()
    tokens = []
    for token, tag in pos_tag(word_tokenize(text)):
        pos = tag[0].lower()
        if pos not in ['a', 'r', 'n', 'v']:
            pos = 'n'
        tokens.append(wordnet.lemmatize(token, pos))
    return ' '.join(tokens)


def stemming(text):
    porter = PorterStemmer()
    result = []
    for word in text.split(' '):
        result.append(porter.stem(word))
    return ' '.join(result)

# todo: removing frequent words (word is neutral)
# todo: check if punctuation signs are in connection with the sentiment when dataset is complete
# todo: check if caps lock is in connection with the sentiment when dataset is complete
# todo: counting words and analyzing sentiment when dataset is complete (word has sentiment)
# todo: graphics
# todo: bert, TFIDF, word2vec
def preprocess(data):
    reviews = data['review']
    update_review = []
    stop_words = stopwords.words('english')
    spell = SpellChecker()
    for review in reviews:
        review = str.lower(review)
        review = replace_emoji(review)
        review = replace_special_chars(review)
        review = remove_urls(review)
        all_words = review.split(' ')
        update_list = []
        for word in all_words:
            if word != '' and word not in stop_words:
                word = spell.correction(word)
                update_list.append(word)
        lemmed_review = lemmatization(' '.join(update_list))
        stemmed_review = stemming(lemmed_review)
        update_review.append(stemmed_review)

    data['text'] = update_review
    return data

if __name__ == '__main__':
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    # nltk.download('punkt')
    # nltk.download('stopwords')

    data = load('../data/english-reviews/english-reviews.csv')
    data = preprocess(data)
    # print(data)

    # without using GridSearchCV
    x = data['review']
    y = data['sentiment']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=101)

    Tfidf_vect = TfidfVectorizer(use_idf=False, lowercase=False, sublinear_tf=True, ngram_range=(1, 2))
    Tfidf_vect.fit(data['review'])
    x_train_tfidf = Tfidf_vect.transform(x_train)
    x_test_tfidf = Tfidf_vect.transform(x_test)

    model = SVC()
    model.fit(x_train_tfidf, y_train)
    predictions = model.predict(x_test_tfidf)
    print(classification_report(y_test, predictions))
    print("accuracy: ", accuracy_score(y_test, predictions))
    #
    # # with GridSearchCV
    # tuned_parameters = {'svm__C': [1, 10, 100], 'svm__kernel': ['sigmoid']}
    # x_train = data['review']
    # y_train = data['sentiment']
    # vectorizer = TfidfVectorizer(use_idf=False, lowercase=False, sublinear_tf=True, ngram_range=(1, 2))
    # pipeline = Pipeline([
    #     ('vec', vectorizer),
    #     ('svm', svm.SVC())
    # ])
    #
    # grid_search = GridSearchCV(pipeline, tuned_parameters, scoring='accuracy', cv=5, error_score='raise', refit=True)
    # grid_search.fit(x_train, y_train)
    # predictions = grid_search.predict(x_train)
    # score = metrics.accuracy_score(y_train, predictions)
    # print(metrics.classification_report(y_train, predictions))

    # pd.options.display.max_colwidth = 500
    # print(data['review'].to_string())
    # data.to_csv('result.csv', sep='|')