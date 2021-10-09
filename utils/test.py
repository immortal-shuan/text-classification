import os
import torch
import argparse
import numpy as np
import time
import multiprocessing as mp
from tqdm import trange, tqdm
from utils.vocab_class import Vocab
from model.bert import model_classification
from model.lstm import TextRCNN
from model.cnn import TextCNN
from sklearn.metrics import accuracy_score, roc_auc_score


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--max_len', default=512)
    arg_parser.add_argument('--model_name', choices=['lstm', 'bert', 'cnn', 'gnn'], default='cnn')
    arg_parser.add_argument('--data_path', default='../data')
    arg_parser.add_argument('--vocab_file', default='../data/vocab.json')
    arg_parser.add_argument('--bert_path', default='F:/pretrain_model/rbtl3')
    arg_parser.add_argument('--output_path', default='../model_output')
    arg_parser.add_argument('--num_class', default=14)
    arg_parser.add_argument('--batch_size', default=256)

    args = arg_parser.parse_args()
    return args


args = init_arg_parser()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
vocab = Vocab(args.vocab_file)


def model_load():
    pretrained_model_name = 'model_{}.pth'.format(args.model_name)
    pretrained_model_path = os.path.join(args.output_path, pretrained_model_name)
    pretrained_model = torch.load(pretrained_model_path)
    if args.model_name == 'bert':
        model = model_classification(args=args)
    elif args.model_name == 'lstm':
        model = TextRCNN(embed_dim=64, hidden_dim=128, dropout=0.5, num_class=args.num_class)
    elif args.model_name == 'cnn':
        model = TextCNN(128, args.max_len, 16, 0.5, args.num_class, vocab_size=5049)
    model.load_state_dict(pretrained_model)
    model = model.to(device)
    return model


model = model_load()


def test_gpu(batch_data):
    batch_data = convertText2ids(batch_data)
    text_input, label = Batch(batch_data, args)
    out = model(text_input, None)
    pred_prob = torch.softmax(out, -1).cpu().tolist()
    pred = out.argmax(dim=-1).cpu().tolist()
    res = []
    assert len(label) == len(pred_prob) == len(pred)
    for i in range(len(label)):
        res.append([label[i], pred_prob[i], pred[i]])
    return res


def convertText2ids(data):
    res = []
    for sample in data:
        text_input = [vocab.word_2_idx(word) for word in sample[0][0:1020:2]]
        res.append([text_input, sample[-1]])
    return res


def test_cpu(sample):
    text = sample[0]
    text_input = [vocab.word_2_idx(word) for word in text[0:1020:2]]
    text_input = torch.tensor([text_input], dtype=torch.long).to(device)
    out = model(text_input, None)
    pred_prob = torch.softmax(out, -1).cpu().tolist()[0]
    pred = out.argmax(dim=-1).cpu().tolist()[0]
    return [pred_prob, pred, sample[1]]


def getResult(data):
    if device == torch.device('cpu'):
        # 查看cpu核心数
        num_cores = int(mp.cpu_count())
        # processes=1 指定使用多少核心进行计算
        with mp.Pool(processes=1) as p:
            result = p.map(test_cpu, [sample for sample in data])
    else:
        result = []
        data_len = len(data)
        with torch.no_grad():
            for m in trange(0, data_len, args.batch_size):
                if m + args.batch_size < data_len:
                    dev_batch_data = data[m: m + args.batch_size]
                else:
                    dev_batch_data = data[m: data_len]
                result.extend(test_gpu(dev_batch_data))
    return result


def Batch(data, args):
    text_input = []
    label = []
    for sample in data:
        text_input.append(sample[0])
        label.append(sample[-1])

    text_input_ = batch_pad(text_input, args, pad=0)
    return text_input_, label


def batch_pad(batch_data, args, pad=0):
    max_len = args.max_len
    out = []
    for line in batch_data:
        if len(line) < max_len:
            out.append(line + [pad] * (max_len - len(line)))
        else:
            out.append(line[:args.max_len])
    return torch.tensor(out, dtype=torch.long).to(device)


# 文本数据的读取方式
def textReadOneEach(path):
    data = []
    with open(path, encoding='utf-8', mode='r') as f:
        for line in f:
            data.append(eval(line))
        f.close()
    return data


if __name__ == '__main__':
    test_path = os.path.join(args.data_path, 'THUCNews_test.txt')
    test_data = textReadOneEach(test_path)
    start_time = time.time()
    print(start_time)
    out = getResult(test_data)
    end_time = time.time()
    print(end_time-start_time)
