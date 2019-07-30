import pickle
import os.path
import numpy as np
from collections import defaultdict
from math import log

from pyspark.sql import functions as F
from mySkipGram import mSkipGram
import pyspark
from pyspark.sql import SQLContext
spark = pyspark.SparkContext(master='local[*]', appName="myAppName")
sql = SQLContext(spark)

'''
TRAIN THE THREE GROUPS OF I2B2 NER
'''

'''
TODO:
+ figure out the format to input embeddings to spark-nlp.
+ tf-idf for term selection in groups.
+ optimization to avoid unnecessary distance computation.
+ change beta according to tf-idf scores of terms.

'''

sents = []
nSents = 200000

path_prefix = '../../auxdata/umls/'
constraints_file = 'i2b2_ner_constraints.p'
semantic_types = 'SemanticTypes_2018AB.csv'

# categories as defined by i2b2 NER
# https://www.i2b2.org/NLP/Relations/assets/Concept%20Annotation%20Guideline.pdf

# e.g., 'Neoplastic process' -> T191
desc2st = sql.read.option('delimiter', '|').csv(path_prefix + semantic_types)
desc2st = dict([(x[2].lower(), x[1]) for x in desc2st.collect()])

# should go to a separate JSON
# Medical Problems
problems = set(['pathologic function', 'disease or syndrome', 'mental or behavioral dysfunction',
                'cell or molecular dysfunction', 'congenital abnormality', 'acquired abnormality',
                'injury or poisoning', 'anatomical abnormality', 'neoplastic process', 'virus', 'bacterium',
                'sign or symptom'])
problems = {desc2st[desc] for desc in problems}

# Medical Treatments
treatments = set(['therapeutic or preventive procedure', 'medical device', 'organic chemical', 'pharmacologic substance',
                  'biomedical or dental material', 'antibiotic', 'clinical drug', 'drug delivery device'])
treatments = {desc2st[desc] for desc in treatments}

# Medical Tests
tests = set(['laboratory procedure', 'diagnostic procedure'])
tests = {desc2st[desc] for desc in tests}

classes = {'problems' : problems, 'treatments': treatments, 'tests':tests}
classes_counts = defaultdict(float)

# load dataset - PubMed/i2b2
def sents():
    with open('i2b2_pubmed_200k.txt') as f:
        for line in f.readlines():
            splits = line.split()
            cleaned = []
            for t in splits:
                if t.endswith('.') or t.endswith(','):
                    cleaned.append(t[:-1])
                    cleaned.append(t[-1])
                else:
                    cleaned.append(t)
            yield cleaned

# load the costraints
def clean_tokenize(term):
    term = term.lower().replace('(', ' ')
    term = term.replace(')', ' ')
    return term.split()


if not os.path.isfile(constraints_file):
    mrconso = sql.read.option('delimiter', '|').csv(path_prefix + 'MRCONSO.RRF.aa')
    mrconso = mrconso.filter(mrconso._c1 == 'ENG')
    mrsty = sql.read.option('delimiter', '|').csv(path_prefix + 'MRSTY.RRF')
    terms = mrconso.join(mrsty, mrconso._c0 == mrsty._c0).groupby(mrconso._c14).agg(F.collect_set(mrsty._c1)).collect()

    # term -> set of semantic types
    categories = {pair['_c14']: pair['collect_set(_c1)'] for pair in terms}
    token_categories = dict()
    token_counts = {k:defaultdict(int) for k in classes}

    total = 0
    for term, stys in categories.items():
        # this one is a one or more from {tests, treatments, etc}
        main_classes = set([c for c, v in classes.items() if len(v.intersection(stys)) > 0])

        for main_class in main_classes:
            # clean term(tokenize, normalize) and perform metrics
            terms = clean_tokenize(term)
            for t in terms:
                token_categories[t] = main_classes
                token_counts[main_class][t] += 1.0
            classes_counts[main_class] += 1.0

    for main_class in main_classes:
        for t in token_counts[main_class]:
            tf = token_counts[main_class][t] / classes_counts[main_class]
            idf = len(categories) / sum([token_counts[c][t] for c in classes])

            # override with tfidf
            token_counts[main_class][t] = tf * log(idf)

    # keep only best scored terms on each class
    for c in token_counts:
        top = np.quantile(np.array(list(token_counts[c].values())), .75)
        token_counts[c] = {k:v for k,v in token_counts[c].items() if v > top}

    with open(constraints_file, 'wb') as f:
        pickle.dump(categories, f)

else:
    with open(constraints_file, 'rb') as f:
        categories = pickle.load(f)

# comment/uncomment next 4 lines
sg = pickle.load(open('skipGram_173ksent.p', 'rb'))
sg.applyConstraints = True
sg.beta = 0.001
sg.continue_train(sents)

#sg = mSkipGram(sents, categories)
#pickle.dump(sg, open("skipgram.p", "wb"))
print(list(sg.most_similar('cell', 5)))

# sg.wEmbed['ship']
