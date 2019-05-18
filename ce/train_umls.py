import pickle
import os.path
from pyspark.sql import functions as F
from mySkipGram import mSkipGram

sents = []
nSents = 200000

path_prefix = '../../auxdata/umls/'
constraints_file = 'umls_constraints.p'

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
if not os.path.isfile(constraints_file):
    mrconso = spark.read.option('delimiter', '|').csv(path_prefix + 'MRCONSO.RRF.aa')
    mrconso = mrconso.filter(mrconso._c1 == 'ENG')
    mrsty = spark.read.option('delimiter', '|').csv(path_prefix + 'MRSTY.RRF')
    terms = mrconso.join(mrsty, mrconso._c0 == mrsty._c0).groupby(mrconso._c14).agg(F.collect_set(mrsty._c1)).collect()

    # semantic type -> set of word ids
    categories = {pair['_c14']: pair['collect_set(_c1)'] for pair in terms}

    # keep only single word terms
    categories = {k: set(categories[k]) for k in categories if len(k.split(" ")) == 1}
    with open(constraints_file, 'wb') as f:
        pickle.dump(categories, f)

else:
    with open(constraints_file, 'rb') as f:
        categories = pickle.load(f)

# keep viruses and bacteria only
categories = {k.lower():v for k,v in categories.items() if 'T005' in v or 'T007' in v}

# comment/uncomment next 4 lines
sg = pickle.load(open('skipGram_173ksent.p', 'rb'))
sg.applyConstraints = True
sg.beta = 0.01
sg.continue_train(sents)

#sg = mSkipGram(sents, categories)
#pickle.dump(sg, open("skipgram.p", "wb"))
print(list(sg.most_similar('cell', 5)))

# sg.wEmbed['ship']
