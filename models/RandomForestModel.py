from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from preprocessing import SerbianPreprocessing as sp
from preprocessing import EnglishPreprocessing as ep
import pandas as pd
import pickle
import os


# from sklearn.metrics import plot_confusion_matrix
# import matplotlib.pyplot as plt


def tfidf_initialization(X):
    tfidf_vectorizer = TfidfVectorizer(use_idf=False, lowercase=False, sublinear_tf=True, ngram_range=(1, 2))
    tfidf_vectorizer.fit(X)
    return tfidf_vectorizer


def train(data):
    # without using GridSearchCV
    x = data['review']
    y = data['sentiment']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=101)

    tfidf_vectorizer = tfidf_initialization(data['review'])
    x_train_tfidf = tfidf_vectorizer.transform(x_train)
    x_test_tfidf = tfidf_vectorizer.transform(x_test)

    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        oob_score=True,
        n_jobs=-1,
    )
    model.fit(x_train_tfidf, y_train)
    predictions = model.predict(x_test_tfidf)
    print(classification_report(y_test, predictions))

    # plot_confusion_matrix(
    #     forest,
    #     test_tokenized,
    #     Yt,
    #     display_labels=["Negative", "Neutral", "Positive"],
    #     normalize=None
    # )
    # plt.show()

    return model, tfidf_vectorizer

def train_e_grid_search():
    # with GridSearchCV
    best_result = 0
    all_results = []

#    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
#    'max_features': ['auto', 'sqrt'],
#    'min_samples_leaf': [1, 2, 4],
#    'min_samples_split': [2, 5, 10],
#    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]


    for i in range(1):
        tuned_parameters = {'rfc__max_depth': [10, 20],
                            'rfc__max_features': ['auto', 'sqrt'],
                            'rfc__min_samples_leaf': [1, 2],
                            'rfc__min_samples_split': [2, 5],
                            'rfc__n_estimators': [200, 400]
                            }
        data = pd.read_csv('../data/english-reviews/english-reviews.csv', delimiter='|', encoding='utf-8',
                           names=['sentiment', 'review'], skiprows=1)
        data = ep.df_preprocess(data)
        #
        train, test = train_test_split(data, test_size=0.2, shuffle=True, stratify=data['sentiment'])
        X_train = train['review'].values
        X_test = test['review'].values
        y_train = train['sentiment']
        y_test = test['sentiment']

        vectorizer = TfidfVectorizer(use_idf=False, lowercase=False, sublinear_tf=True, ngram_range=(1, 2))

        pipeline = Pipeline([
            ('vec', vectorizer),
            ('rfc', RandomForestClassifier())
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
            file_name = os.path.join('rfm_models_english_ngram_1_2', 'rfm_' + str(i) + '_' + str(round(best_result, 5)))
            file_name_description = os.path.join('rfm_models_english_ngram_1_2', 'rfm_' + str(i) + '_' + str(
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
    data = pd.read_csv('../data/english-reviews/english-reviews-test.csv', delimiter='|', encoding='utf-8',
                       names=['sentiment', 'review'], skiprows=1)
    original_reviews = data['review'].values

    data = sp.df_preprocess(data)
    # data = data.sample(frac=1)
    # print(data)
    X_test = data['review']
    print(X_test.values)
    y_test = data['sentiment']

    pipeline = Pipeline([
        ('vec', TfidfVectorizer(use_idf=False, lowercase=False, sublinear_tf=True, ngram_range=(1, 2))),
        ('rfm', RandomForestClassifier())
    ])

    file_name = os.path.join('rfm_models_english_ngram_1_2', 'rfm_0_0.77128')
    with open(file_name, 'rb') as f:
        loaded_best_params = pickle.load(f)

    pipeline.set_params(**loaded_best_params)
    predictions = pipeline.predict(X_test)
    print(predictions)
    score = accuracy_score(y_test, predictions)
    print(score)

    result_df = pd.DataFrame(list(zip(predictions, y_test, original_reviews)), columns=['prediction', 'original', 'review'])
    file_name = os.path.join('rfm_result_english', 'result.csv')
    result_df.to_csv(file_name, encoding='utf-8', sep='|', index=False)


def train_s_grid_search():
    # with GridSearchCV
    best_result = 0
    all_results = []

    for i in range(1):
        tuned_parameters = {'rfc__max_depth': [10, 20],
                            'rfc__max_features': ['auto', 'sqrt'],
                            'rfc__min_samples_leaf': [1, 2],
                            'rfc__min_samples_split': [2, 5],
                            'rfc__n_estimators': [200, 400]
                            }
        data = pd.read_csv('../data/serbian-reviews/serbian-reviews.csv', delimiter='|', encoding='utf-8',
                           names=['sentiment', 'review'], skiprows=1)
        data = sp.df_preprocess(data)

        train, test = train_test_split(data, test_size=0.2, shuffle=True, stratify=data['sentiment'])
        X_train = train['review'].values
        X_test = test['review'].values
        y_train = train['sentiment']
        y_test = test['sentiment']

        vectorizer = TfidfVectorizer(use_idf=False, lowercase=False, sublinear_tf=True, ngram_range=(1, 2))

        rfc = RandomForestClassifier()

        pipeline = Pipeline([
            ('vec', vectorizer),
            ('rfc', rfc)
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
            file_name = os.path.join('rfm_models_serbian_ngram_1_2', 'rfm_' + str(i) + '_' + str(round(best_result, 5)))
            file_name_description = os.path.join('rfm_models_serbian_ngram_1_2', 'rfm_' + str(i) + '_' + str(round(best_result, 5)) + '_description.txt')
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
    data = pd.read_csv('../data/serbian-reviews/serbian-reviews-test.csv', delimiter='|', encoding='utf-8',
                       names=['sentiment', 'review'], skiprows=1)
    original_reviews = data['review'].values

    data = sp.df_preprocess(data)
    # data = data.sample(frac=1)
    # print(data)
    X_test = data['review']
    print(X_test.values)
    y_test = data['sentiment']

    pipeline = Pipeline([
        ('vec', TfidfVectorizer(use_idf=False, lowercase=False, sublinear_tf=True, ngram_range=(1, 2))),
        ('rfm', RandomForestClassifier())
    ])

    file_name = os.path.join('rfm_models_serbian_ngram_1_2', 'rfm_0_0.88845')
    with open(file_name, 'rb') as f:
        loaded_best_params = pickle.load(f)

    pipeline.set_params(**loaded_best_params)
    predictions = pipeline.predict(X_test)
    print(predictions)
    score = accuracy_score(y_test, predictions)
    print(score)

    result_df = pd.DataFrame(list(zip(predictions, y_test, original_reviews)), columns=['prediction', 'original', 'review'])
    file_name = os.path.join('rfm_result_serbian', 'result.csv')
    result_df.to_csv(file_name, encoding='utf-8', sep='|', index=False)


if __name__ == '__main__':
    # train_s_grid_search()
    # test_s_grid_search()
     train_e_grid_search()
    # test_e_grid_search()
