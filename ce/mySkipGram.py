# date: Aug 2017
# brief: implementation of word2vec (skip-gram with negative sampling)

from __future__ import division
import numpy as np
import pickle
import random
from numpy.linalg import norm
from collections import defaultdict

def expit(ndarray):
    return np.exp(-np.logaddexp(0, -ndarray))

class mSkipGram:

    flatten = lambda l: [item for sublist in l for item in sublist]

    def __init__(self, sentences, categories, nEmbed=100, negativeRate=5, stepsize=.025, winSize=2, minCount=1, epochs=2):
        # TODO: filtering categories twice?
        self.whitelist = set(['T005', 'T007'])
        self.applyConstraints = True
        self.labels = np.zeros(negativeRate + 1)
        self.labels[0] = 1.
        self.stepsize = stepsize
        self.winSize = winSize
        self.negativeRate = negativeRate
        self.categories = categories
        self.nEmbed = nEmbed
        # influence of the constraints in cost
        self.beta = 0.01

        self.vocab = defaultdict(int)
        #self.trainset = sentences()

        for sentence in sentences():
            for word in sentence:
                self.vocab[word] += 1

        words = list(self.vocab.keys())
        for word in words:
            if self.vocab[word] < minCount:
                self.vocab.pop(word)
        self.nVocab = len(self.vocab)

        self.wEmbed = np.zeros((self.nVocab, nEmbed))
        self.cEmbed = (np.random.rand(self.nVocab, nEmbed) - .5) / self.nEmbed
        self.w2id = {}

        for idx, word in enumerate(self.vocab):
            self.w2id[word] = idx

        # filter constraints - keep only the ones we do have in the vocabulary
        keys = set(self.w2id.keys())
        self.id2types = {self.w2id[word]: categories[word] for word in categories if word in keys}
        self.type2ids = defaultdict(set)

        for k, types in self.id2types.items():
            for type in types:
                self.type2ids[type].add(k)

        self.make_cum_table()
        self.accLoss = 0.0
        self.trainWords = 0
        self.loss = []

        self.zeroVector = np.zeros(nEmbed, dtype=np.float)

        for iter in range(epochs):
            # probably the shuffling not necessary if the DS is big enough
            #random.shuffle(self.trainset)
            self.train(sentences())
            self.stepsize = max(.0001, self.stepsize - .005)

    def continue_train(self, sentences, epochs=2):
        for iter in range(epochs):
            self.train(sentences())
            self.stepsize = max(.0001, self.stepsize - .005)

    def make_cum_table(self, power=0.75, domain=2 ** 31 - 1):
        """
            copied from gensim
        """
        self.cum_table = np.zeros(self.nVocab, dtype=np.uint32)
        # compute sum of all power
        train_words_pow = 0.0
        Z = 0.
        for count in self.vocab.values():
            Z += count ** power
        cumulative = 0.0
        for word, wcount in self.vocab.items():
            widx = self.w2id[word]
            cumulative += wcount ** power
            self.cum_table[widx] = round(cumulative / Z * domain)
        if len(self.cum_table) > 0:
            assert self.cum_table[-1] == domain, (self.cum_table[-1], domain, Z)

    def sample(self, omit):
        res = set()
        while len(res) != self.negativeRate:
            ridx = self.cum_table.searchsorted(np.random.randint(self.cum_table[-1]))
            if ridx not in omit:
                res.add(ridx)
        return list(res)

    def trainWord(self, wordId, contextId, negativeIds):
        word = self.wEmbed[wordId]

        # contexts (positive id + negative ids)
        cIds = [contextId] + negativeIds
        contexts = self.cEmbed[cIds]

        prod = np.dot(word, contexts.T) # array of scalars, x*y, x*z1, x*z2 ...
        prodexp = expit(prod)  # expit(x) = 1/(1+exp(-x))

        gradient = (self.labels - prodexp) * self.stepsize
        #self.cEmbed[cIds] += np.outer(gradient, word) - self.beta * np.apply_along_axis(self.dD, axis=0, arr=cIds)

        self.cEmbed[cIds] += np.outer(gradient, word)
        word += np.dot(gradient, contexts)

        if self.applyConstraints:
            self.cEmbed[cIds] -= self.beta * np.array(list(map(self.dD, cIds)))
            word -= self.beta * self.dD(wordId)

        # for logging
        self.accLoss -= sum(np.log(expit(-1 * prod[1:]))) + np.log(expit(prod[0]))

    def train(self, trainset):
        for counter, sentence in enumerate(trainset):
            sentence = list(filter(lambda w: w in self.vocab, sentence))

            for wpos, word in enumerate(sentence):
                wIdx = self.w2id[word]
                winsize = np.random.randint(self.winSize) + 1
                start = max(0, wpos - winsize)
                end = min(wpos + winsize + 1, len(sentence))

                # print '%d +/- %d = [%d,%d]'%(wpos, winsize, start, end)
                for context_word in sentence[start:end]:
                    ctxtId = self.w2id[context_word]
                    if ctxtId == wIdx: continue
                    negativeIds = self.sample({wIdx, ctxtId})
                    self.trainWord(wIdx, ctxtId, negativeIds)
                    self.trainWords += 1

            if counter % 1000 == 0:
                print('> training sum(np.log(expit(-1 * prod[1:]))) + np.log(expit(prod[0]))%d sentences' % counter)
                if self.trainWords == 0:
                    print(sentence)
                self.loss.append(self.accLoss / self.trainWords)
                self.trainWords = 0
                self.accLoss = 0.0
            #if counter == 1000:
                #self.measure_category_similarity()
                #break

    def most_similar(self, word, k=10):
        from sklearn.preprocessing import normalize
        xnorm = normalize(self.cEmbed)
        sim = np.dot(xnorm, xnorm[self.w2id[word]])
        idx = np.argsort(sim)[::-1]
        topk = filter(lambda i: i != self.w2id[word], idx[:k + 1])
        topk = list(topk)

        return zip([list(self.vocab.keys())[i] for i in topk], list(sim[topk]))

    def __getitem__(self, word):
        return self.cEmbed[self.w2id[word]]

    def dD(self, i):
        '''
        compute the derivative of the constraints' cost function
        '''
        # if 'i' not in any semantic type(kj is empty) return the '0 vector'
        result = np.zeros(self.nEmbed)
        if i not in self.id2types:
            return result

        # find the semantic types where this word belongs
        pos_types = [type for type in self.id2types[i]]
        pos_types = [t for t in pos_types if t in self.whitelist]

        # merge all ids coming from all previous types
        merged_pos_wids = set(mSkipGram.flatten([self.type2ids[t] for t in pos_types]))

        # find the semantic types where this word does not belong
        neg_types = [type for type in self.type2ids if type not in self.id2types[i]]
        neg_types = [t for t in neg_types if t in self.whitelist]
        merged_neg_wids = set(mSkipGram.flatten([self.type2ids[t] for t in neg_types]))

        for k in merged_neg_wids:
            if not np.any(self.wEmbed[k]):
                continue
            for j in merged_pos_wids:
                if not np.any(self.wEmbed[j]):
                    continue
                result += self.dhinge(self.sij(self.wEmbed[i], self.wEmbed[k])- self.sij(self.wEmbed[i], self.wEmbed[j]))\
                          * (self.dsij(self.wEmbed[i], self.wEmbed[k]) - self.dsij(self.wEmbed[i], self.wEmbed[j]))

        return result

    def delta(self, i, j, t):
        if t == i or t ==j:
            return 1.0
        return 0.0

    def sij(self, wi, wj):

        '''
        computes the cosine similarity between wi and wj
        '''
        norm_prod = norm(wi) * norm(wj)
        if norm_prod == 0.0:
            return self.zeroVector

        dotprod = np.dot(wi, wj)
        return dotprod / norm_prod

    def dsij(self, wi, wj):

        '''
        computes the derivative of the cosine similarity between wi and wj with respect to wi

        '''
        sij = self.sij(wi, wj)
        norm_prod = norm(wi) * norm(wj)
        if norm_prod == 0.0:
            return self.zeroVector

        return (sij / (norm(wi) ** 2)) * wi + wj / norm_prod
   
    def dhinge(self, vector):
        '''
        computes the derivative of the hinge function on the input vector
        '''
        return (vector > 0.0) * vector

    def measure_category_similarity(self):
        ''' measures how close (on average) to each other vectors of the same category are '''
        avg_dist = 0.0
        for category in self.whitelist:
            category_wids = list(self.type2ids[category])
            same_category = self.wEmbed[category_wids]
            for vector in same_category:
                avg_dist += sum(np.dot(vector, same_category.T)) / 100.0

        return avg_dist

    def measure_intercategory_similarity(self):
        ''' measures how close to each other vectors of different category are '''
        avg_dist = 0.0
        for this_category in self.whitelist:
            for other_category in [c for c in self.whitelist if other_category != this_category]:
                other_category_wids = list(self.type2ids[other_category])
                other_category_vectors = self.wEmbed[other_category_wids]
                this_category_wids = list(self.type2ids[this_category])
                this_category_vectors = self.wEmbed[this_category_wids]
                for vector in this_category_vectors:
                    avg_dist += sum(np.dot(vector, other_category_vectors.T)) / 100.0

        return avg_dist

if __name__ == '__main__':
    sents = []
    nSents = 200000

    filename = '../data/unlabeledTrainData.tsv'
    reader = KaggleDataAccess()
    sents = list(reader.loadDataset(filename))

    sg = mSkipGram(sents)
    pickle.dump(sg, open("myskipgram1.p", "wb"))
    print(list(sg.most_similar('mountain', 4)))

    #sg.wEmbed['ship']
