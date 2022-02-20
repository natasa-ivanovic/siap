import pandas as pd
from spellchecker import SpellChecker

from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import word_tokenize,pos_tag
from nltk.stem import PorterStemmer

'''
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
'''


emoji_dict = {
    'excellent': [':-)', ':)', ':-d', ':d', 'xd', '😀', '😁', '😃', '😄', '😆', '😉', '😊', '😋', '😎', '😍', '😘', '🥰', '😗', '😙', '😚', '☺', '🙂', '🤗', '🤩', '😌', '😛', '🙃', '😏', '🤤', '🙄', '5+', '10+', '5.00', '5/5', '10/10', '👍', '👌'],
    'great': ['<3', '❤', '🧡', '💛', '💚', '💙', '💜', '🤎', '🖤', '🤍', '❣', '💕', '💞', '💓', '💗', '💖', '💘', '💝', '💟', '💌'],
    'bad': [':(', ':-(', '</3', '💔', '🤔', '🤨', '😐', '😑', '😶', '😥', '🤐', '😫', '🥱', '😴', '😒', '😓', '😔', '😕', '🤑', '☹', '🙁', '😖', '😞', '😟', '😢', '😭', '😦', '😧', '😨', '😩', '😬', '😰', '🥵', '😱', '😡', '🥺', '😣', '😠', '🤬', '🤮', '🤒', '🤢', '🤧', '🥶', '😤'],
    'ok': ['😂', '🤣'],
    '': [';)', ';-)', 'D:']
}


def load():
    return pd.read_csv('../data/english-reviews/english-reviews.csv', delimiter='|', encoding='utf-8', names=['title', 'text', 'rating', 'sentiment'], skiprows=1)


def remove_whitespace(text):
    return " ".join(text.split())


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


def lemming(text):
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
def preprocess(review):
    stop_words = stopwords.words('english')
    spell = SpellChecker()
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
    lemmed_review = lemming(' '.join(update_list))
    stemmed_review = stemming(lemmed_review)

    return stemmed_review


def df_preprocess(data):
    reviews = data['review']
    update_review = []
    for review in reviews:
        u_review = preprocess(review)
        update_review.append(u_review)
    data['review'] = update_review
    return data
