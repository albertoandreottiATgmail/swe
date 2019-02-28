# author: mgalle
# date: Aug 2017
# brief: implementation of word2vec (skip-gram with negative sampling)

from __future__ import division
from functools import reduce
from collections import defaultdict
import numpy as np
import pickle
import random
from numpy.linalg import norm
from collections import defaultdict

def expit(ndarray):
    return np.exp(-np.logaddexp(0, -ndarray))

class mSkipGram:

    flatten = lambda l: [item for sublist in l for item in sublist]

    def __init__(self, sentences, categories, nEmbed=100, negativeRate=5, stepsize=.025, winSize=2, minCount=1, epochs=5):
        self.labels = np.zeros(negativeRate + 1)
        self.labels[0] = 1.
        self.stepsize = stepsize
        self.winSize = winSize
        self.negativeRate = negativeRate
        self.categories = categories
        self.nEmbed = nEmbed
        # influence of the constraints in cost
        self.beta = 0.2

        self.vocab = defaultdict(int)

        for sentence in sentences:
            for word in sentence:
                self.vocab[word] += 1

        words = list(self.vocab.keys())
        for word in words:
            if self.vocab[word] < minCount:
                self.vocab.pop(word)
        self.nVocab = len(self.vocab)
        self.trainset = sentences

        self.wEmbed = np.zeros((self.nVocab, nEmbed))
        self.cEmbed = (np.random.rand(self.nVocab, nEmbed) - .5) / self.nEmbed
        self.w2id = {}

        for idx, word in enumerate(self.vocab):
            self.w2id[word] = idx

        # filter constraints
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

        for iter in range(epochs):
            random.shuffle(self.trainset)
            self.train()
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
        self.cEmbed[cIds] += np.outer(gradient, word) - self.beta * np.array(list(map(self.dD, cIds)))
        word += np.dot(gradient, contexts) - self.beta * self.dD(wordId)

        # for logging
        self.accLoss -= sum(np.log(expit(-1 * prod[1:]))) + np.log(expit(prod[0]))

    def train(self):
        for counter, sentence in enumerate(self.trainset):
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
                    # print
            if counter % 1000 == 0:
                print('> training %d of %d' % (counter, len(self.trainset)))
                if self.trainWords == 0:
                    print(sentence)
                self.loss.append(self.accLoss / self.trainWords)
                self.trainWords = 0
                #self.accLoss = 0.0

    def most_similar(self, word, k=10):
        xnorm = 1 # normalize(self.cEmbed), disabled by now
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

        if i not in self.constraints:
            return result

        # find the semantic types where this word belongs
        pos_types = [type for type in self.id2types[i]]
        # merge all ids coming from all previous types
        merged_pos_wids = set(flatten([self.type2ids[t] for t in pos_types]))

        # find the semantic group(s) where this word does not belong
        neg_types = [self.type2ids[type] for type in self.type2ids if type not in self.id2types[i]]
        merged_neg_wids = set(flatten([self.type2ids[t] for t in neg_types]))

        kj = zip(merged_neg_wids, merged_pos_wids)
        for k, j in kj:
            result += self.dhinge(self.dsij(i, k) - self.dsij(i, j))

        return result

    def sij(self, wi, wj):

        '''
        computes the cosine similarity between wi and wj
        '''
        norm_prod = norm(wi) * norm(wj)
        return np.dot(wi, wj) / norm_prod

    def dsij(self, wi, wj):

        '''
        computes the derivative of the cosine similarity between wi and wj with respect to wi

        '''
        sij = self.sij(wi, wj)
        return (sij / norm(wi) ** 2) * wi + wj / (norm(wi) * norm(wj))
   

    def dhinge(self, vector):
        '''
        computes the derivative of the hinge function on the input vector
        '''
        return (vector > 0.0) * vector

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
