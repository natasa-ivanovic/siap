from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


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
