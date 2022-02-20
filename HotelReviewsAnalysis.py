from preprocessing import EnglishPreprocessing as ep
from preprocessing import SerbianPreprocessing as sp
from language_detection import LanguageDetection as ld
from models import SVMModel

import pandas as pd

if __name__ == '__main__':
    print('>> Data loading...')
    en_data = pd.read_csv('./data/english-reviews/english-reviews.csv', delimiter='|', encoding='utf-8',
                          names=['title', 'review', 'rating', 'sentiment'], skiprows=1)
    srb_data = pd.read_csv('./data/serbian-reviews/serbian-reviews.csv', delimiter='|', encoding='utf-8',
                           names=['title', 'review', 'sentiment'], skiprows=1)
    print('>> Data successfully loaded...')

    print('>> English reviews preprocessing...')
    en_data_p = ep.df_preprocess(en_data)
    print('>> Serbian reviews preprocessing...')
    srb_data_p = sp.df_preprocess(srb_data)

    print('>> Training phase for english reviews...')
    en_svm, en_vectorizer = SVMModel.train(en_data_p)
    print('>> Training phase for serbian reviews...')
    srb_svm, srb_vectorizer = SVMModel.train(srb_data_p)

    print('>> Training phase for language detection model...')
    lp_model, vectorizer = ld.train(srb_data, en_data)

    # test = 'Hello my dear friend I really enjoyed this hotel.'
    test = 'Zaista sam se dobro provela u ovom baš lepom hotelu.'
    # test = 'Ovo je užasno do bola!! Negativno iskustvo! Sobe smrde, ništa nije kao na slici!!! Odvratno je i gadi mi se.'
    # test = 'I disliked the food and the room was too small. Did not enjoy it at all. AWFUL'
    language = ld.predict(test, vectorizer, lp_model)
    if language == 0:
        # serbian
        data = sp.preprocess(test)
        data_s = ' '.join([str(elem) for elem in data])
        vectorized = srb_vectorizer.transform([data_s])
        print('Prediction: ', srb_svm.predict(vectorized)) #vraca listu predictiona umesto jednu?
    else:
        # english
        data = ep.preprocess(test)
        vectorized = en_vectorizer.transform([data])
        print('Prediction: ', en_svm.predict(vectorized)) #vraca jednu predikciju

