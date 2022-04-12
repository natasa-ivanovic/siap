# http://agateteam.org/spacynerannotate/
# https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy

import spacy
import random
import pathlib
import pandas as pd
from spacy.training import Example
from spacy.scorer import Scorer

NEW_LABELS = ['PERSONELL', 'WEATHER', 'FOOD', 'HYGIENE', 'FURNITURE', 'LOCATION', 'PRICE']
ITERATIONS = 50


def load(path):
    return pd.read_csv(path, delimiter='|', encoding='utf-8', names=['title', 'review', 'rating', 'sentiment'], skiprows=1)


def evaluate(ner_model, data):
    scorer = Scorer()
    examples = []
    for t, annotations in data:
        doc = ner_model.make_doc(t)
        example = Example.from_dict(doc, annotations)
        example.predicted = ner_model(example.predicted)
        examples.append(example)
    scores = scorer.score(examples)
    return scores


def train():
    data = load('./data/english-reviews/english-reviews.csv')

    # create blank spacy model with ner pipe
    nlp = spacy.blank('en')
    nlp.add_pipe('ner')
    nlp.begin_training()
    ner = nlp.get_pipe('ner')

    # load data
    TRAIN_DATA = []
    with open('data/ner/ner_english', 'r') as file:
        data = file.read().replace('\n', '')
        TRAIN_DATA = eval(data)
    # print(TRAIN_DATA)

    # add labels to ner pipe
    for label in NEW_LABELS:
        ner.add_label(label)

    optimizer = nlp.resume_training()
    move_names = list(ner.move_names)

    pipe_exceptions = ['ner', 'trf_wordpiecer', 'trf_tok2vec']
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    print(nlp.get_pipe('ner').labels)

    # training the model
    with nlp.disable_pipes(*unaffected_pipes):
        for iteration in range(ITERATIONS):
            print("Iteration: ", iteration)
            # shuufling examples before every iteration
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            for batch in spacy.util.minibatch(TRAIN_DATA, size=20):
                for t, annotations in batch:
                    doc = nlp.make_doc(t)
                    example = Example.from_dict(doc, annotations)
                    sgd = optimizer
                    nlp.update(
                        [example], losses=losses, drop=0.3
                    )
                # print("Losses", losses)
            print(losses)

    # save and load model
    nlp.to_disk(str(pathlib.Path().resolve()) + '\spacy_model\\')

    nlp = spacy.load(str(pathlib.Path().resolve()) + '\spacy_model\\')
    results = evaluate(nlp, TRAIN_DATA)

    # ents_p, ents_r, ents_f are the precision, recall and fscore for the NER task
    print(results)
    print('PRECISION: ', results['ents_p'])
    print('RECALL: ', results['ents_r'])
    print('FSCORE: ', results['ents_f'])

    with open('data/ner/ner_english_results_train', 'w') as results_file:
        results_file.write(str(results))
        results_file.write('\nPRECISION: ' + str(results['ents_p']))
        results_file.write('\nRECALL: ' + str(results['ents_r']))
        results_file.write('\nFSCORE: ' + str(results['ents_f']))
        for sentence in TRAIN_DATA:
            doc = nlp(sentence[0])
            results_file.write('\nEntities ')
            list_results = ([(ent.text, ent.label_) for ent in doc.ents])
            results_file.write(str(list_results))


def test():
    TEST_DATA = []
    with open('data/ner/ner_english_test', 'r') as file:
        test_data = file.read().replace('\n', '')
        TEST_DATA = eval(test_data)
    print(TEST_DATA)

    nlp = spacy.load(str(pathlib.Path().resolve()) + '\spacy_model\\')
    results = evaluate(nlp, TEST_DATA)
    # ents_p, ents_r, ents_f are the precision, recall and fscore for the NER task
    print(results)
    print('PRECISION: ', results['ents_p'])
    print('RECALL: ', results['ents_r'])
    print('FSCORE: ', results['ents_f'])

    with open('data/ner/ner_english_results_test', 'w') as results_file:
        results_file.write(str(results))
        results_file.write('\nPRECISION: ' + str(results['ents_p']))
        results_file.write('\nRECALL: ' + str(results['ents_r']))
        results_file.write('\nFSCORE: ' + str(results['ents_f']))
        for sentence in TEST_DATA:
            doc = nlp(sentence[0])
            results_file.write('\nEntities ')
            list_results = ([(ent.text, ent.label_) for ent in doc.ents])
            results_file.write(str(list_results))


if __name__ == '__main__':
    train()


