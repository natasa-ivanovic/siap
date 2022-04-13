import pandas as pd
import re

TRAIN_PATH_S = '../data/serbian-reviews/serbian-reviews.csv'
TRAIN_PATH_E = '../data/english-reviews/english-reviews.csv'


def replace_chars(review):
    for c in '0123456789.,!?-:();*="\'':
        review = review.replace(c, '')
    return review


if __name__ == '__main__':
    data = pd.read_csv(TRAIN_PATH_E, sep='|')
    reviews = data['review']
    all_words = dict()
    in_letters = 'čćđžš'
    out_letters = 'ccdzs'
    remove_characters = ',!?-:();"\'0123456789'
    trans_tab = str.maketrans(in_letters, out_letters)

    sent_dict = {-1: [], 1: []}

    for review, sentiment in zip(reviews, data['sentiment']):
        review = replace_chars(review)
        words = review.split(' ')
        for word in words:
            processed_word = str(word).translate(trans_tab)
            if processed_word == processed_word.upper() and processed_word != 'I' and processed_word != '':
                if sentiment == 1:
                    sent_dict[1].append(processed_word)
                else:
                    sent_dict[-1].append(processed_word)

    #res = sorted(sent_dict, key=lambda key:all_words[key], reverse=True)
    print(sent_dict)
    for el in sent_dict:
        print(el, ':\t', len(sent_dict[el]))