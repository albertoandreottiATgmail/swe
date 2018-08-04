# load dataset
# 0 negative, 1 positive

class KaggleDataAccess(object):
    def __init__(self):
        self.mapper = {'.': ' .', ',':' ,', "'": " ' ", '\\"':' " ', "\\'": " ' "}

    def normalize(self, text):
        normalized = text
        mapper = self.mapper
        for c in mapper:
            normalized = normalized.replace(c, mapper[c])
        return normalized

    def loadDataset(self, path):
        '''
        :return: iterator over a dataset
        '''
        with open(path) as f:
            header_len = len(next(f).split('\t'))  # ommit header
            text_location = 1 if header_len == 2 else 2
            has_label = header_len == 3
            for line in f:
                values = line.split('\t')
                chunks = [word.strip() for word in self.normalize(values[text_location]).split()]
                if len(chunks) < 10:
                    continue

                if has_label:
                    # this is labeled data, yield the text and the label
                    yield (chunks, values[1])
                else:
                    # this is unlabeled data, yield only the text
                    yield chunks
