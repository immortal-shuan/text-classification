import time
from utils.vocab_class import Vocab
import multiprocessing as mp
from transformers import BertTokenizer


train_path = '../data/THUCNews_train.txt'
bert_path = 'F:/pretrain_model/rbt3'
vocab_file = '../data/vocab.json'
tokenizer = BertTokenizer.from_pretrained(bert_path)
vocab = Vocab(vocab_file)


# 将文本转化成input_ids
def convert_text_to_ids(sample):
    input_info = tokenizer.encode_plus(sample[0][:510], add_special_tokens=True)
    input_ids = input_info['input_ids']
    attention_mask = input_info['attention_mask']
    return [input_ids, attention_mask, sample[-1]]


def convert_text2ids(sample):
    text = sample[0]
    text_input = [vocab.word_2_idx(word) for word in text[:510]]
    text_mask = [1]*len(text_input)
    return [text_input, text_mask, sample[-1]]


# 对文本数据读取
def textReadOneEach(path):
    data = []
    with open(path, encoding='utf-8', mode='r') as f:
        for line in f:
            data.append(eval(line))
        f.close()
    return data


# 并行处理数据
def dataPro(data):
    # 查看cpu核心数
    num_cores = int(mp.cpu_count())
    # processes=1 指定使用多少核心进行计算
    with mp.Pool(processes=1) as p:
        result = p.map(convert_text2ids, [sample for sample in data])
    return result


if __name__ == '__main__':
    # train_data = textReadOneEach(bert_path)

    # pool = mp.Pool(processes=8)
    train_data = textReadOneEach(train_path)
    start_time = time.time()
    # result = pool.apply_async(textReadOneEach, (train_path,))
    # result = pool.apply_async(f, (10,))
    new_data = dataPro(train_data)
    end_time = time.time()
    print(end_time-start_time)