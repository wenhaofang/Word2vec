import os
import math
import random

from loaders.Huffman import HuffmanTree

class SG_HS_Loader():
    def __init__(self,
        min_freq,
        max_numb,
        max_window_size,
        neg_sample_size, # no use
        batch_size,
        file_path
    ):
        self.texts = self.read_srfile(file_path)
        self.vocab = self.build_vocab(self.texts, min_freq, max_numb)
        self.ids   = self.encode_text(self.texts)
        self.datas = self.build_datas(self.ids, max_window_size, neg_sample_size)

        self.batch_size = batch_size

    def read_srfile(self, file_path):
        # text8 and ptb dataset has been processed. Just use it.
        with open(file_path, 'r', encoding = 'utf-8') as txt_file:
            texts = txt_file.readlines()
            texts = [[word for word in text.split() if len(word) > 0] for text in texts]
        return texts

    def build_vocab(self, texts, min_freq, max_numb):
        counter = {}
        for text in texts:
            for word in text:
                counter[word] = counter.get(word, 0) + 1

        corpus = sorted(counter.items(), key = lambda item: item[1], reverse = True)
        corpus = [word for idx, (word, count) in enumerate(corpus) if count >= min_freq and idx < max_numb]

        vocab = {}
        vocab['id2word'] = { idx: word for idx, word in enumerate(corpus) }
        vocab['word2id'] = { word: idx for idx, word in enumerate(corpus) }
        vocab['word2freq'] = { word: freq for word, freq in counter.items() if word in corpus }

        assert len(vocab['id2word']) == len(vocab['word2id']) == len(vocab['word2freq'])

        return vocab

    def subsampling(self, texts):
        def discard(word):
            word_freq = self.vocab['word2freq'].get(word)
            words_num = sum([len(text) for text in texts])
            return (
                random.uniform(0, 1) < 1 - math.sqrt(1e-4 / (word_freq / words_num))
            )

        texts = [[word for word in text if word in self.vocab['word2freq']] for text in texts]
        texts = [[word for word in text if not discard(word)] for text in texts]

        return texts

    def encode_text(self, texts):
        return [[self.vocab['word2id'].get(word) for word in text] for text in self.subsampling(texts)]

    def hufsampling(self, word_id, mode):
        if (
            not hasattr(self, 'huffman_pos_paths') or
            not hasattr(self, 'huffman_neg_paths')
        ):
            id2word , word2freq = self.vocab['id2word'], self.vocab['word2freq']
            id2freq = dict([(word_id, word2freq.get(id2word.get(word_id))) for word_id in id2word])
            huffman_tree = HuffmanTree( id2freq )
            self.huffman_pos_paths, self.huffman_neg_paths = huffman_tree.get_pos_and_neg_path()

        return (
            self.huffman_pos_paths[word_id] if mode == 'pos' else \
            self.huffman_neg_paths[word_id] if mode == 'neg' else \
            []
        )

    def build_datas(self, ids_list, max_window_size, neg_sample_size):
        datas = []
        for ids in ids_list:
            if len(ids) < 2:
                continue

            for center_idx in range(len(ids)):
                center_word = ids[center_idx]

                window_size = random.randint(1, max_window_size)
                background_range = range(
                    max(0 , center_idx - window_size),
                    min(len(ids) , center_idx + window_size + 1)
                )
                background_words = [
                    ids[background_idx] for background_idx in background_range if background_idx != center_idx
                ]

                pos_words = [self.hufsampling(background_word, mode = 'pos') for background_word in background_words]
                neg_words = [self.hufsampling(background_word, mode = 'neg') for background_word in background_words]

                pos_words = [word for word_lost in pos_words for word in word_lost]
                neg_words = [word for word_lost in neg_words for word in word_lost]

                datas.append((
                    center_word,
                    pos_words,
                    neg_words,
                ))

        return datas

    def get_vocab_self(self):
        return self.vocab

    def get_vocab_size(self):
        return len(self.vocab['id2word'])

    def reset(self):
        random.shuffle(self.datas)
        self.cur_batch = 0
        self.batch_num = (len(self.datas) // self.batch_size) + (0 if len(self.datas) % self.batch_size == 0 else 1)

    def __iter__(self):
        return self

    def __next__(self):
        if  self.cur_batch < self.batch_num:
            batch_data = self.datas[
                self.cur_batch * self.batch_size :
                self.cur_batch * self.batch_size + self.batch_size
            ]
            self.cur_batch += 1

            max_len = max([len(pos_words) + len(neg_words) for _, pos_words, neg_words in batch_data])
            batch_src_words, batch_trg_words, wmasks, labels = [], [], [], []
            for src_word, pos_words, neg_words in batch_data:
                pos_len = len(pos_words)
                neg_len = len(neg_words)
                cur_len = pos_len + neg_len
                batch_src_words += [src_word]
                batch_trg_words += [pos_words + neg_words + [0] * (max_len - cur_len)]
                wmasks += [[1] * cur_len + [0] * (max_len - cur_len)]
                labels += [[1] * pos_len + [0] * (max_len - pos_len)]

            return (
                batch_src_words, batch_trg_words, wmasks, labels
            )

        else:
            raise StopIteration

def get_loader(option):

    if option.dataset_name == 'text8':
        file_name = 'text8'

    if option.dataset_name == 'ptb':
        file_name = 'ptb.train.txt'

    return SG_HS_Loader(
        option.min_freq,
        option.max_numb,
        option.max_window_size,
        option.neg_sample_size,
        option.batch_size,
        os.path.join(option.dataset_path, option.dataset_name, file_name)
    )

if __name__ == '__main__':
    from utils.parser import get_parser

    parser = get_parser()
    option = parser.parse_args()

    loader = get_loader(option)

    loader.reset()
    for mini_batch in loader:
        src_words, trg_words, wmasks, labels = mini_batch
        print(type(src_words), len(src_words)) # List, batch_size
        print(type(trg_words), len(trg_words), len(trg_words[0])) # Nested List, batch_size, seq_len
        print(type(wmasks), len(wmasks), len(wmasks[0])) # Nested List, batch_size, seq_len
        print(type(labels), len(labels), len(labels[0])) # Nested List, batch_size, seq_len
        break
