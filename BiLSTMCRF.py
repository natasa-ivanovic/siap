# https://blog.dominodatalab.com/named-entity-recognition-ner-challenges-and-model
# https://confusedcoders.com/data-science/deep-learning/how-to-build-deep-neural-network-for-custom-ner-with-kera

import operator
# pip install git+https://www.github.com/keras-team/keras-contrib.git
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from future.utils import iteritems
from keras import layers
from keras.models import Model
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow_addons.layers import CRF

NEW_LABELS = ['PERSONNEL', 'WEATHER', 'FOOD', 'HYGIENE', 'LOCATION', 'FURNITURE']
TAGS = ['B-PERSONNEL', 'B-WEATHER', 'B-FOOD', 'B-HYGIENE', 'B-LOCATION', 'B-FURNITURE', 'I-PERSONNEL', 'I-WEATHER',
        'I-FOOD', 'I-HYGIENE', 'I-LOCATION', 'I-FURNITURE', 'O']
text = 'Smeštaj uredan i čist, Olja kao domaćin izvanredna, niko nam još nije bio na usluzi kao ona. Konačno smo i za u buduće našli smeštaj koji nam odgovara.'

FOOD_SENTENCES = ['Makaroni su italijanska hrana .',
                  'Nudle su veoma popularne .',
                  'Pohovana piletina se jede u cijelom svijetu .',
                  'Lazanje su Garfildova omiljena hrana .',
                  'Suši je poznato i skupo jelo u Japanu .',
                  'Unagi je morska hrana koja potiče iz Japana .',
                  'Svi vole jesti bijelu čokoladu .',
                  'Pica je najbolja brza hrana .',
                  'Djeca ne vole jesti špinat .'
                  ]
FOOD_DATA = [
    [('Makaroni', 'B-FOOD'), ('su', 'O'), ('italijanska', 'O'), ('hrana', 'O'), ('.', 'O')],
    [('Nudle', 'B-FOOD'), ('su', 'O'), ('veoma', 'O'), ('popularne', 'O'), ('.', 'O')],
    [('Pohovana', 'B-FOOD'), ('piletina', 'I-FOOD'), ('se', 'O'), ('jede', 'O'), ('u', 'O'), ('cijelom', 'O'),
     ('svijetu', 'O'), ('.', 'O')],
    [('Lazanje', 'B-FOOD'), ('su', 'O'), ('Garfildova', 'O'), ('omiljena', 'O'), ('hrana', 'O'), ('.', 'O')],
    [('Suši', 'B-FOOD'), ('je', 'O'), ('poznato', 'O'), ('i', 'O'), ('skupo', 'O'), ('jelo', 'O'), ('u', 'O'),
     ('Japanu', 'O'), ('.', 'O')],
    [('Unagi', 'B-FOOD'), ('je', 'O'), ('morska', 'O'), ('hrana', 'O'), ('koja', 'O'), ('potiče', 'O'), ('iz', 'O'),
     ('Japana', 'O'), ('.', 'O')],
    [('Svi', 'O'), ('vole', 'O'), ('jesti', 'O'), ('bijelu', 'B-FOOD'), ('čokoladu', 'I-FOOD'), ('.', 'O')],
    [('Pica', 'B-FOOD'), ('je', 'O'), ('najbolja', 'O'), ('brza', 'O'), ('hrana', 'O'), ('.', 'O')],
    [('Djeca', 'O'), ('ne', 'O'), ('vole', 'O'), ('jesti', 'O'), ('špinat', 'B-FOOD'), ('.', 'O')],
]

FOOD_TAGS = ['B-FOOD', 'I-FOOD', 'O']

if __name__ == '__main__':

    # convert data to ner data
    # data_for_ner = []
    # data = pd.read_csv('serbian reviews/serbian-reviews.csv', delimiter='|', encoding='utf-8', names=['title', 'review', 'sentiment'], skiprows=1)
    # data = data.head(5)
    # for index, row in data.iterrows():
    #    text = row['review']
    #    for c in text:
    #        if c in '.,!?-:();*="\'\\/#':
    #            text = text.replace(c, " " + c + " ")
    #    for word in text.split():
    #        data_for_ner.append([index, word, 'O'])
    # df_for_ner = pd.DataFrame(data_for_ner, columns=['sentence', 'word', 'label'])
    # df_for_ner.to_csv('ner_serbian.csv', sep='|')

    # load ner data
    TRAIN_DATA = []
    ner_data = pd.read_csv('data/ner/ner_serbian.csv', delimiter='|', encoding='utf-8',
                           names=['sentence', 'word', 'label'], skiprows=1)
    sentences_indices = np.unique(ner_data['sentence'])
    for i in sentences_indices:
        sentence_data = ner_data[ner_data['sentence'] == i]
        sentence = []
        for index, row in sentence_data.iterrows():
            sentence.append((row['word'], row['label']))
        TRAIN_DATA.append(sentence)
    # print(TRAIN_DATA)

    TRAIN_DATA = FOOD_DATA  # test
    print(TRAIN_DATA)

    words = np.unique(ner_data['word'])
    words = sum(map(lambda x: x.split(), FOOD_SENTENCES), [])  # test
    n_words = len(words)
    tags = TAGS
    tags = FOOD_TAGS  # test
    n_tags = len(tags)

    print('ALL WORDS: ', words)

    word2idx = {w: i + 1 for i, w in enumerate(np.unique(words))}
    word2idx["--UNKNOWN_WORD--"] = 0
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2word = {v: k for k, v in iteritems(word2idx)}
    idx2tag = {v: k for k, v in iteritems(tag2idx)}

    for k, v in sorted(word2idx.items(), key=operator.itemgetter(1)):
        print(k, v)

    X = [[word[0] for word in sentence] for sentence in TRAIN_DATA]
    y = [[word[1] for word in sentence] for sentence in TRAIN_DATA]
    print("Sentence 1:", X[0])
    print("Labels 1:", y[0])

    X = [[word2idx[word] for word in sentence] for sentence in X]
    y = [[tag2idx[tag] for tag in sentence] for sentence in y]
    print("Sentence 1:", X[0])
    print("Labels 1:", y[0])

    MAX_SENTENCE = max([len(s) for s in TRAIN_DATA])

    X = [sentence + [1] * (MAX_SENTENCE - len(sentence)) for sentence in X]
    y = [sentence + [0] * (MAX_SENTENCE - len(sentence)) for sentence in y]
    # X = [sentence + [word2idx["--PADDING--"]] * (MAX_SENTENCE - len(sentence)) for sentence in X]
    # y = [sentence + [tag2idx["--PADDING--"]] * (MAX_SENTENCE - len(sentence)) for sentence in y]
    print("Sentence 1:", X[0])
    print("Labels 1:", y[0])

    TAG_COUNT = len(tag2idx)
    y = [np.eye(TAG_COUNT)[sentence] for sentence in y]
    print("Sentence 1:", X[0])
    print("Labels 1:", y[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    print("Number of sentences in the training dataset: {}".format(len(X_train)))
    print("Number of sentences in the test dataset : {}".format(len(X_test)))

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    WORD_COUNT = len(idx2word)
    DENSE_EMBEDDING = 32
    LSTM_UNITS = 50
    LSTM_DROPOUT = 0.1
    DENSE_UNITS = 100
    BATCH_SIZE = 256
    MAX_EPOCHS = 5

    input_layer = tf.keras.layers.Input(shape=(MAX_SENTENCE,))

    print('WORD COUNT: ', WORD_COUNT)
    print('MAX_SENTENCE: ', MAX_SENTENCE)

    model = tf.keras.layers.Embedding(WORD_COUNT + 1, DENSE_EMBEDDING, embeddings_initializer="uniform",
                                      input_length=MAX_SENTENCE)(
        input_layer)

    model = tf.keras.layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, recurrent_dropout=LSTM_DROPOUT, return_sequences=True))(model)

    model = tf.keras.layers.TimeDistributed(layers.Dense(DENSE_UNITS, activation="relu"))(model)

    crf_layer = CRF(units=TAG_COUNT)
    # output_layer = crf_layer(model)
    _, output_layer, _, _ = crf_layer(model)

    ner_model = Model(input_layer, output_layer)

    loss = 'categorical_crossentropy'
    acc_metric = 'accuracy'  # MeanSquaredError
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    ner_model.compile(optimizer=opt, loss=loss, metrics=[acc_metric])
    ner_model.summary()

    print('TRAINING DATA: ', X_train)
    print('INDICES: ', word2idx)

    history = ner_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, validation_split=0.1, verbose=2)

    ner_model.save(str(pathlib.Path().resolve()) + '\\bilstmcrf_model\\')
    ner_model = tf.keras.models.load_model(str(pathlib.Path().resolve()) + '\\bilstmcrf_model\\')

    y_pred = ner_model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=2)
    y_test = np.argmax(y_test, axis=2)
    accuracy = (y_pred == y_test).mean()
    print("Accuracy: {:.4f}".format(accuracy))


    def tag_conf_matrix(cm, tagid):
        # tag_name = idx2tag[tagid]
        tag_name = tagid
        tagid = tag2idx[tag_name]
        print("Tag name: {}".format(tag_name))
        print(cm[tagid])
        tn, fp, fn, tp = cm[tagid].ravel()
        tag_acc = (tp + tn) / (tn + fp + fn + tp)
        print("Tag accuracy: {:.3f} \n".format(tag_acc))


    matrix = multilabel_confusion_matrix(y_test.flatten(), y_pred.flatten())

    # tag_conf_matrix(matrix, 'B-FOOD')
    # tag_conf_matrix(matrix, 'I-FOOD')

    sentence = FOOD_SENTENCES[0].split()

    # padded_sentence = sentence + [word2idx["--PADDING--"]] * (MAX_SENTENCE - len(sentence))
    padded_sentence = sentence + [1] * (MAX_SENTENCE - len(sentence))
    padded_sentence = [word2idx.get(w, 0) for w in padded_sentence]

    pred = ner_model.predict(np.array([padded_sentence]))
    pred = np.argmax(pred, axis=-1)

    retval = ""
    for w, p in zip(sentence, pred[0]):
        retval = retval + "{:15}: {:5}".format(w, idx2tag[p])
        " + \n"
    print(retval)

    # Plot the graph
    plt.style.use('ggplot')


    def plot_history(history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

    # plot_history(history)
