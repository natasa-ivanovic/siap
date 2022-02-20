from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def tfidf_initialization(X):
    tfidf_vectorizer = TfidfVectorizer(use_idf=False, lowercase=False, sublinear_tf=True, ngram_range=(1, 2))
    tfidf_vectorizer.fit(X)
    return tfidf_vectorizer


def train(data):
    # without using GridSearchCV
    x = data['review']
    y = data['sentiment']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=101)

    tfidf_vectorizer = tfidf_initialization(data['review'])
    x_train_tfidf = tfidf_vectorizer.transform(x_train)
    x_test_tfidf = tfidf_vectorizer.transform(x_test)

    model = SVC()
    model.fit(x_train_tfidf, y_train)
    predictions = model.predict(x_test_tfidf)
    print(classification_report(y_test, predictions))
    # print("accuracy: ", accuracy_score(y_test, predictions))

    return model, tfidf_vectorizer

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
