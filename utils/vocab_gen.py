import os
import json
import collections


# 生成词表文件，
# 此表格式{word：word_num, ...}
# word为词，word_num为词次数
def gene_word_freq(train_dir, vocab_file):
    word_freq = collections.Counter()

    data = textReadOneEach(train_dir)
    for sample in data:
        text = sample[0]
        word_freq.update(text)

    word_freq = word_freq.most_common(len(word_freq))
    word_freq = dict(word_freq)

    vocab_file = os.path.join(vocab_file)

    print(len(word_freq))
    print(list(word_freq.items())[:20])

    with open(vocab_file, "w", encoding='utf-8') as f:
        json.dump(word_freq, f)
        f.close()

    return word_freq


def textReadOneEach(path):
    data = []
    with open(path, encoding='utf-8', mode='r') as f:
        for line in f:
            data.append(eval(line))
        f.close()
    return data


if __name__ == "__main__":
    train_path = '../data/THUCNews_train.txt'
    vocab_file = '../data/vocab.json'
    word_freq = gene_word_freq(train_path, vocab_file)