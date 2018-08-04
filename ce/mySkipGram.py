# author: mgalle
# date: Aug 2017
# brief: implementation of word2vec (skip-gram with negative sampling)


from __future__ import division
import numpy as np
# from scipy.special import expit
from collections import defaultdict
# from sklearn.preprocessing import normalize
import pickle
import random

from data_access import KaggleDataAccess


def expit(ndarray):
    return np.exp(-np.logaddexp(0, -ndarray))

class mSkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, stepsize=.025, winSize=2, minCount=1, epochs=5):
        self.labels = np.zeros(negativeRate + 1)
        self.labels[0] = 1.
        self.stepsize = stepsize
        self.winSize = winSize
        self.negativeRate = negativeRate
        self.nEmbed = nEmbed

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
        self.cEmbed[cIds] += np.outer(gradient, word)
        word += np.dot(gradient, contexts)

        # for logging
        self.accLoss -= sum(np.log(expit(-1 * prod[1:]))) + np.log(expit(prod[0]))

    def train(self):
        for counter, sentence in enumerate(self.trainset):
            sentence = list(filter(lambda word: word in self.vocab, sentence))

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
                self.loss.append(self.accLoss / self.trainWords)
                self.trainWords = 0
                self.accLoss = 0.

   def most_similar(self, word, k=10):
        xnorm = 1 # normalize(self.cEmbed), disabled by now
        sim = np.dot(xnorm, xnorm[self.w2id[word]])
        idx = np.argsort(sim)[::-1]
        topk = filter(lambda i: i != self.w2id[word], idx[:k + 1])
        topk = list(topk)

        return zip([list(self.vocab.keys())[i] for i in topk], list(sim[topk]))

    def __getitem__(self, word):
        return self.cEmbed[self.w2id[word]]


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
