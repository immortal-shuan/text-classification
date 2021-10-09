import json


class Vocab(object):
    def __init__(self, vocab_file, vob_num=5049):

        self.vob_num = vob_num
        # 建立{字符：索引，...}字典
        self.word_to_idx = {}
        # 字符列表
        self.idx_to_word = []
        # 词表特殊字符分别是[空白填充， 不存在，句子开头字符，句子结尾字符]
        SPECIAL_TOKEN = ['<pad>', '<unk>', '<start>', '<stop>']
        # idx表示索引，token表示词的字符串
        for idx, token in enumerate(SPECIAL_TOKEN):
            self.word_to_idx[token] = idx
            self.idx_to_word.append(token)

        # 将四种特殊字符表示出来，方便后面使用
        self.pad_idx = 0
        self.pad_token = SPECIAL_TOKEN[self.pad_idx]
        self.unk_idx = 1
        self.unk_token = SPECIAL_TOKEN[self.unk_idx]
        self.start_idx = 2
        self.start_token = SPECIAL_TOKEN[self.start_idx]
        self.stop_idx = 3
        self.stop_token = SPECIAL_TOKEN[self.stop_idx]

        # 读取生成的词表文件
        with open(vocab_file, "r", encoding='utf-8') as f:
            word_freq = json.load(f)
        f.close()

        vocab_len = len(self.idx_to_word)
        # word_freq为字典格式{词:词的个数，...}
        # enumerate读取只能读取字典的keys即词，无法读取后面的个数
        # 即token只表示词
        for i, token in enumerate(word_freq):
            idx = vocab_len + i
            if idx >= self.vob_num:
                break
            self.word_to_idx[token] = idx
            self.idx_to_word.append(token)

    # 返回字典word_to_idx中word的索引，若不存在word，返回self.unk_idx = 1
    def word_2_idx(self, word):
        return self.word_to_idx.get(word, self.unk_idx)

    # 若存在，返回索引所在的单词，返回self.unk_idx = 1
    def idx_2_word(self, idx):
        if (idx >= 0) and (idx < self.vob_num):
            return self.idx_to_word[idx]
        else:
            return self.unk_idx

    # 返回词表词的数量
    def get_vob_size(self):
        return self.vob_num


if __name__ == "__main__":
    vocab_file = '../data/vocab.json'
    vocab = Vocab(vocab_file)