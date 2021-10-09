import os
import argparse
import random
from tqdm import tqdm



def init_data_path_arg():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--label2num', default={
    '财经': 250, '彩票': 50, '房产': 120, '股票': 900, '家居': 200, '教育': 250,
    '科技': 800, '社会': 300, '时尚': 100, '时政': 400, '体育': 880, '星座': 50,
    '游戏': 150, '娱乐': 550
    })

    arg_parser.add_argument('--data_path', default='F:/NLPdata/THUCNews/textData')
    arg_parser.add_argument('--save_path', default='../data')

    args = arg_parser.parse_args()
    return args


def textReadOneEach(path):
    data = []
    with open(path, encoding='utf-8', mode='r') as f:
        for line in f:
            sample = eval(line)
            sample_text = sample[0]
            if 800 <= len(sample_text) <= 1500:
                data.append(sample)
        f.close()
    return data


def textAvgLen(data):
    total_len = 0
    for sample in data:
        text_len = len(sample[0])
        total_len += text_len
    return total_len / len(data)


def randomSelect(data, num):
    return random.sample(data, num)


def SaveToTxt(data, path):
    with open(path, mode='a', encoding='utf-8') as f:
        for sample in data:
            f.write(str(sample) + '\n')
        f.close()


args = init_data_path_arg()

train_data = []
test_data = []
for label_name in args.label2num.keys():
    path = os.path.join(args.data_path, '{}.txt'.format(label_name))
    data = textReadOneEach(path)
    temp_data = randomSelect(data, args.label2num[label_name]*2)
    random.shuffle(temp_data)
    temp_train = temp_data[:len(temp_data)//2]
    temp_test = temp_data[len(temp_data)//2:]
    assert len(temp_train) == len(temp_test)
    train_data.extend(temp_train)
    test_data.extend(temp_test)

save_train = '../data/THUCNews_train.txt'
save_test = '../data/THUCNews_test.txt'
random.shuffle(train_data)
random.shuffle(test_data)
print(len(train_data), len(test_data))
SaveToTxt(train_data, save_train)
SaveToTxt(test_data, save_test)








