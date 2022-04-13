from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from preprocessing import SerbianPreprocessing as sp
from preprocessing import EnglishPreprocessing as ep
import pandas as pd
import joblib
import pickle
import os

TRAIN_S = '../data/serbian-reviews/serbian-reviews.csv'
TRAIN_E = '../data/english-reviews/english-reviews.csv'
TEST_S = '../data/serbian-reviews/serbian-reviews-test.csv'
TEST_E = '../data/english-reviews/english-reviews-test.csv'


def tfidf_initialization(X):
    tfidf_vectorizer = TfidfVectorizer(use_idf=False, lowercase=False, sublinear_tf=True, ngram_range=(1, 2))
    tfidf_vectorizer.fit(X)
    return tfidf_vectorizer


def train(data):
    # without using GridSearchCV
    x = data['review']
    y = data['sentiment']

    print('Negative sentiment num: ', y.values.flatten().tolist().count(-1))
    print('Positive sentiment num: ', y.values.flatten().tolist().count(1))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.10, random_state=101)

    tfidf_vectorizer = tfidf_initialization(data['review'])
    x_train_tfidf = tfidf_vectorizer.transform(x_train)
    x_test_tfidf = tfidf_vectorizer.transform(x_test)

    model = SVC()
    model.fit(x_train_tfidf, y_train)
    predictions = model.predict(x_test_tfidf)
    print(classification_report(y_test, predictions))
    # print("accuracy: ", accuracy_score(y_test, predictions))

    return model, tfidf_vectorizer


def train_e_grid_search():
    # with GridSearchCV
    best_result = 0
    all_results = []

    for i in range(10):
        # tuned_parameters = {'svm__kernel': ['linear'], 'svm__C': [0.1, 1, 10, 100, 200, 1000]}
        # tuned_parameters = {'svm__kernel': ['rbf'], 'svm__C': [0.1, 1, 10, 100, 200, 1000],
        #                     'svm__gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
        # tuned_parameters = {'svm__kernel': ['poly'], 'svm__C': [0.1, 1, 10, 100, 200, 1000],
        #                     'svm__gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
        # tuned_parameters = {'svm__kernel': ['sigmoid'], 'svm__C': [0.1, 1, 10, 100, 200, 1000],
        #                     'svm__gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

        tuned_parameters = {'svm__kernel': ['rbf', 'poly', 'sigmoid'], 'svm__C': [0.1, 1, 10, 100, 200, 1000],
                            'svm__gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'vec__ngram_range': [(1, 1), (1, 2), (1, 3)]}

        data = pd.read_csv(TRAIN_E, delimiter='|', encoding='utf-8',
                           names=['sentiment', 'review'], skiprows=1)
        data = ep.df_preprocess(data)
        #
        train, test = train_test_split(data, test_size=0.2, shuffle=True, stratify=data['sentiment'])
        X_train = train['review'].values
        X_test = test['review'].values
        y_train = train['sentiment']
        y_test = test['sentiment']

        vectorizer = TfidfVectorizer(use_idf=False, lowercase=False, sublinear_tf=True, ngram_range=(1, 3))

        pipeline = Pipeline([
            ('vec', vectorizer),
            ('svm', svm.SVC())
        ])

        grid_search = GridSearchCV(pipeline, tuned_parameters, scoring='accuracy', cv=5, error_score='raise',
                                   refit=True, verbose=10)
        grid_search.fit(X_train, y_train)
        predictions = grid_search.predict(X_test)
        score = accuracy_score(y_test, predictions)
        all_results.append(score)
        print('Iteration', i, 'score -', score)
        cr = classification_report(y_test, predictions)
        print(cr)

        if score > best_result:
            best_result = score
            file_name = os.path.join('svm_models_english_ngram_1_1', 'svm_' + str(i) + '_' + str(round(best_result, 5)))
            file_name_description = os.path.join('svm_models_english_ngram_1_1', 'svm_' + str(i) + '_' + str(
                round(best_result, 5)) + '_description.txt')
            # joblib.dump(grid_search.best_estimator_, file_name)
            with open(file_name, 'wb') as f:
                pickle.dump(grid_search.best_estimator_.get_params(), f)
            # joblib.dump(grid_search.best_estimator_.get_params(), file_name)
            f = open(file_name_description, 'w')
            f.write('Best parameters after tuning\n')
            f.write(str(grid_search.best_params_))
            f.write('\n\n')
            f.write('\nModel after parameter tuning\n')
            f.write(str(grid_search.best_estimator_))
            f.write('\n\n')
            f.write('\nBest score\n')
            f.write(str(grid_search.best_score_))
            f.write('\n\n')
            f.write('\nClassification report\n')
            f.write(cr)
            f.close()
        print('Best score: ', best_result)
        print('Worst score: ', min(all_results))


def test_e_grid_search():
    data = pd.read_csv(TEST_E, delimiter='|', encoding='utf-8',
                       names=['sentiment', 'review'], skiprows=1)
    original_reviews = data['review'].values

    data = sp.df_preprocess(data)
    # data = data.sample(frac=1)
    # print(data)
    X_test = data['review']
    print(X_test.values)
    y_test = data['sentiment']

    pipeline = Pipeline([
        ('vec', TfidfVectorizer(use_idf=False, lowercase=False, sublinear_tf=True, ngram_range=(1, 3))),
        ('svm', svm.SVC(kernel='linear'))
    ])

    file_name = os.path.join('svm_models_english_ngram_1_1', 'svm_0_0.85638')
    with open(file_name, 'rb') as f:
        loaded_best_params = pickle.load(f)

    pipeline.set_params(**loaded_best_params)
    predictions = pipeline.predict(X_test)
    print(predictions)
    score = accuracy_score(y_test, predictions)
    print(score)

    result_df = pd.DataFrame(list(zip(predictions, y_test, original_reviews)), columns=['prediction', 'original', 'review'])
    file_name = os.path.join('svm_result_english', 'result.csv')
    result_df.to_csv(file_name, encoding='utf-8', sep='|', index=False)


def train_s_grid_search():
    # with GridSearchCV
    best_result = 0
    all_results = []

    for i in range(50):
        # tuned_parameters = {'svm__kernel': ['linear'], 'svm__C': [0.1, 1, 10, 100, 200, 1000]}
        tuned_parameters = {'svm__kernel': ['rbf', 'poly', 'sigmoid'], 'svm__C': [0.1, 1, 10, 100, 200, 1000],
                            'svm__gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'vec__ngram_range': [(1, 1), (1, 2), (1, 3)]}
        # tuned_parameters = {'svm__kernel': ['poly'], 'svm__C': [0.1, 1, 10, 100, 200, 1000],
        #                     'svm__gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
        # tuned_parameters = {'svm__kernel': ['sigmoid'], 'svm__C': [0.1, 1, 10, 100, 200, 1000],
        #                     'svm__gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
        data = pd.read_csv(TRAIN_S, delimiter='|', encoding='utf-8',
                           names=['sentiment', 'review'], skiprows=1)
        data = sp.df_preprocess(data)

        train, test = train_test_split(data, test_size=0.2, shuffle=True, stratify=data['sentiment'])
        X_train = train['review'].values
        X_test = test['review'].values
        y_train = train['sentiment']
        y_test = test['sentiment']

        # vectorizer = TfidfVectorizer(use_idf=False, lowercase=False, sublinear_tf=True, ngram_range=(1, 1))
        vectorizer = TfidfVectorizer(use_idf=False, lowercase=False, sublinear_tf=True)

        pipeline = Pipeline([
            ('vec', vectorizer),
            ('svm', svm.SVC())
        ])

        grid_search = GridSearchCV(pipeline, tuned_parameters, scoring='accuracy', cv=5, error_score='raise', refit=True, verbose=10)
        grid_search.fit(X_train, y_train)
        predictions = grid_search.predict(X_test)
        score = accuracy_score(y_test, predictions)
        all_results.append(score)
        print('Iteration', i, 'score -', score)
        cr = classification_report(y_test, predictions)
        print(cr)

        if score > best_result:
            best_result = score
            file_name = os.path.join('svm_models_serbian_ngram_1_1', 'svm_' + str(i) + '_' + str(round(best_result, 5)))
            file_name_description = os.path.join('svm_models_serbian_ngram_1_1', 'svm_' + str(i) + '_' + str(round(best_result, 5)) + '_description.txt')
            # joblib.dump(grid_search.best_estimator_, file_name)
            with open(file_name, 'wb') as f:
                pickle.dump(grid_search.best_estimator_.get_params(), f)
            # joblib.dump(grid_search.best_estimator_.get_params(), file_name)
            f = open(file_name_description, 'w')
            f.write('Best parameters after tuning\n')
            f.write(str(grid_search.best_params_))
            f.write('\n\n')
            f.write('\nModel after parameter tuning\n')
            f.write(str(grid_search.best_estimator_))
            f.write('\n\n')
            f.write('\nBest score\n')
            f.write(str(grid_search.best_score_))
            f.write('\n\n')
            f.write('\nClassification report\n')
            f.write(cr)
            f.close()
        print('Best score: ', best_result)
        print('Worst score: ', min(all_results))


def test_s_grid_search():
    data = pd.read_csv(TEST_S, delimiter='|', encoding='utf-8',
                       names=['sentiment', 'review'], skiprows=1)
    original_reviews = data['review'].values

    data = sp.df_preprocess(data)
    # data = data.sample(frac=1)
    # print(data)
    X_test = data['review']
    print(X_test.values)
    y_test = data['sentiment']

    pipeline = Pipeline([
        ('vec', TfidfVectorizer(use_idf=False, lowercase=False, sublinear_tf=True, ngram_range=(1, 3))),
        ('svm', svm.SVC(kernel='linear'))
    ])

    file_name = os.path.join('svm_models_serbian_ngram_1_1', 'svm_10_0.95618')
    with open(file_name, 'rb') as f:
        loaded_best_params = pickle.load(f)

    pipeline.set_params(**loaded_best_params)
    predictions = pipeline.predict(X_test)
    print(predictions)
    score = accuracy_score(y_test, predictions)
    print(score)

    result_df = pd.DataFrame(list(zip(predictions, y_test, original_reviews)), columns=['prediction', 'original', 'review'])
    file_name = os.path.join('svm_result_serbian', 'result.csv')
    result_df.to_csv(file_name, encoding='utf-8', sep='|', index=False)


if __name__ == '__main__':
    # train_s_grid_search()
    # test_s_grid_search()
    # train_e_grid_search()
    test_e_grid_search()
