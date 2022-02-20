# http://agateteam.org/spacynerannotate/
# https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy

import spacy
import random
import pathlib
import pandas as pd
#from spacy.util import minibatch, compounding
from spacy.training import Example
from spacy.scorer import Scorer

NEW_LABELS = ['PERSONELL', 'WEATHER', 'FOOD', 'HYGIENE', 'FURNITURE', 'LOCATION']
ITERATIONS = 5


#text = 'Stay at The Whitney to capture a sense of the historry of New Orleans with its actual bank vault door decor and location that is right in the heart of New Orleans great for business but close to the French Quarter for fun Its clean comfortable has amenities and is a great location friendly too Overall good trip Clean and friendly It was mid summer when we stayed and we prefer ice cold room Couldnt get it cool enough Bed was comfy Pillows a bit too thick for us Cold water at faucet was warm to hot Shower temp was perfectly hot Would recommend more mirrors in room A safe would be good addition and fridge Enjoyed continental breakfast Would stay at this beautiful hotel again '
#TRAIN_DATA = [
#(text, {'entities': [(58, 69, 'LOCATION'), (151, 162, 'LOCATION'), (112, 120, 'LOCATION'), (273, 281, 'LOCATION'), (226, 231, 'HYGIENE'), (313, 318, 'HYGIENE'), (282, 290, 'PERSONNEL'), (323, 331, 'PERSONNEL'), (343, 349, 'WEATHER'), (379, 387, 'WEATHER'), (630, 639, 'FOOD')]}),
#              ('China's noodles are very famous', {'entities': [(8,15, 'FOOD')]}),
#              ('Shrimps are famous in China too', {'entities': [(0,7, 'FOOD')]}),
#              ('Lasagna is another classic of Italy', {'entities': [(0,7, 'FOOD')]}),
#              ('Sushi is extemely famous and expensive Japanese dish', {'entities': [(0,5, 'FOOD')]}),
#              ('Unagi is a famous seafood of Japan', {'entities': [(0,5, 'FOOD')]}),
#              ('Tempura , Soba are other famous dishes of Japan', {'entities': [(0,7, 'FOOD')]}),
#              ('Udon is a healthy type of noodles', {'entities': [(0,4, 'ORG')]}),
#              ('Chocolate soufflé is extremely famous french cuisine', {'entities': [(0,17, 'FOOD')]}),
#              ('Flamiche is french pastry', {'entities': [(0,8, 'FOOD')]}),
#              ('Burgers are the most commonly consumed fastfood', {'entities': [(0,7, 'FOOD')]}),
#              ('Burgers are the most commonly consumed fastfood', {'entities': [(0,7, 'FOOD')]}),
#              ('Frenchfries are considered too oily', {'entities': [(0,11, 'FOOD')]})
#]


def load(path):
    return pd.read_csv(path, delimiter='|', encoding='utf-8', names=['title', 'text', 'rating', 'sentiment'], skiprows=1)

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


if __name__ == '__main__':

    data = load('english reviews/reviews.txt')

    #nlp = spacy.load('en_core_web_sm')
    #print(nlp.pipe_names)
    #doc = nlp(text)
    #for ent in doc.ents:
    #    print(ent.text, ent.label_)
    #ner = nlp.get_pipe("ner")

    # create blank spacy model with ner pipe
    nlp = spacy.blank('en')
    nlp.add_pipe('ner')
    nlp.begin_training()
    ner = nlp.get_pipe('ner')

    # load data
    TRAIN_DATA = []
    with open('ner_english', 'r') as file:
        data = file.read().replace('\n', '')
        TRAIN_DATA = eval(data)
    print(TRAIN_DATA)

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
            # shuufling examples before every iteration
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            for batch in spacy.util.minibatch(TRAIN_DATA, size=2):
                for t, annotations in batch:
                    doc = nlp.make_doc(t)
                    example = Example.from_dict(doc, annotations)
                    sgd = optimizer
                    nlp.update(
                        [example], losses=losses, drop=0.3
                    )
                print("Losses", losses)

    # save and load model
    nlp.to_disk(str(pathlib.Path().resolve()) + '\spacy_model\\')
    nlp = spacy.load(str(pathlib.Path().resolve()) + '\spacy_model\\')

    # test model
    #TEST_TEXT = 'Dosa is an extremely famous south Indian dish'
    TEST_TEXT = 'I ate Sushi yesterday. Maggi is a common fast food'

    doc = nlp(TEST_TEXT)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])

    EXAMPLES = [
        (TEST_TEXT,
         {'entities': [(6,11,'FOOD'),(23,28,'FOOD')]})
    ]

    results = evaluate(nlp, EXAMPLES)
    # ents_p, ents_r, ents_f are the precision, recall and fscore for the NER task
    print(results)
    print('PRECISION: ', results['ents_p'])
    print('RECALL: ', results['ents_r'])
    print('FSCORE: ', results['ents_f'])


