import pandas as pd
import re

TRAIN_PATH_S = '../data/serbian-reviews/serbian-reviews.csv'
TRAIN_PATH_E = '../data/english-reviews/english-reviews.csv'

if __name__ == '__main__':
    data = pd.read_csv(TRAIN_PATH_E, sep='|')
    reviews = data['review']
    all_words = dict()
    in_letters = 'čćđžš'
    out_letters = 'ccdzs'
    remove_characters = ',!?-:();"\''
    trans_tab = str.maketrans(in_letters, out_letters)

    sent_dict = {-1: [], 1: []}

    for review, sentiment in zip(reviews, data['sentiment']):
        words = review.split(' ')
        for word in words:
            processed_word = str.lower(word).translate(trans_tab)
            repeating_chars = re.findall(r'((\?){3,})', word)
            if repeating_chars:
                if sentiment == 1:
                    sent_dict[1].append(repeating_chars)
                else:
                    sent_dict[-1].append(repeating_chars)

    #res = sorted(sent_dict, key=lambda key:all_words[key], reverse=True)
    print(sent_dict)
    for el in sent_dict:
        print(el, ':\t', len(sent_dict[el]))