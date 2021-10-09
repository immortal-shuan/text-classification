import os
import json
import torch
import random
import argparse
import numpy as np
from torch.optim import Adam
from tqdm import trange, tqdm
from utils.data_pro import dataPro, textReadOneEach
from transformers import BertTokenizer, BertModel
from model.bert import model_classification
from model.lstm import TextRCNN
from model.cnn import TextCNN
from model.linear import DeepNet
from sklearn.metrics import accuracy_score, roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def setup_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--max_len', default=512)

    arg_parser.add_argument('--stop_num', default=10)
    arg_parser.add_argument('--seed', default=102)
    arg_parser.add_argument('--epoch_num', default=200)
    arg_parser.add_argument('--batch_size', default=256)
    arg_parser.add_argument('--save_model', default=True)
    arg_parser.add_argument('--loss_step', default=1)
    arg_parser.add_argument('--model_name', choices=['lstm', 'bert', 'cnn', 'gnn'], default='lstm')

    arg_parser.add_argument('--data_path', default='../data')
    arg_parser.add_argument('--bert_path', default='F:/pretrain_model/rbtl3')
    arg_parser.add_argument('--output_path', default='../model_output')

    arg_parser.add_argument('--bert_lr', default=2e-5)
    arg_parser.add_argument('--lstm_lr', default=1e-3)
    arg_parser.add_argument('--cnn_lr', default=1e-3)
    arg_parser.add_argument('--dropout', default=0.5)
    arg_parser.add_argument('--bert_dim', default=768)
    arg_parser.add_argument('--num_class', default=14)

    args = arg_parser.parse_args()
    return args


def Batch(data, args):
    text_input = []
    text_mask = []
    label = []
    for sample in data:
        text_input.append(sample[0])
        text_mask.append(sample[1])
        label.append(sample[-1])

    text_input_ = batch_pad(text_input, args, pad=0)
    text_mask_ = batch_pad(text_mask, args, pad=0)
    return text_input_, text_mask_, torch.tensor(label, dtype=torch.long).to(device)


def batch_pad(batch_data, args, pad=0):
    max_len = args.max_len
    out = []
    for line in batch_data:
        if len(line) < max_len:
            out.append(line + [pad] * (max_len - len(line)))
        else:
            out.append(line[:args.max_len])
    return torch.tensor(out, dtype=torch.long).to(device)


def train(train_data, dev_data, model, optimizer, criterion, args):

    train_len = len(train_data)
    model.zero_grad()

    dev_acc = 0.0
    max_acc_index = 0

    for i in range(args.epoch_num):
        random.shuffle(train_data)

        train_step = 1.0
        train_loss = 0.0

        train_preds = []
        train_prob = []
        train_labels = []

        for j in trange(0, train_len, args.batch_size):
            model.train()
            if j + args.batch_size < train_len:
                train_batch_data = train_data[j: j+args.batch_size]
            else:
                train_batch_data = train_data[j: train_len]
            text_input, text_mask, label = Batch(train_batch_data, args)

            out = model(text_input, text_mask)
            loss = criterion(out, label)
            train_loss += loss.item()

            loss = loss / args.loss_step
            loss.backward()

            if int(train_step % args.loss_step) == 0:
                optimizer.step()
                model.zero_grad()

            pred_prob = torch.softmax(out, -1).cpu().tolist()
            pred = out.argmax(dim=-1).cpu().tolist()
            train_preds.extend(pred)
            train_prob.extend(pred_prob)
            train_labels.extend(label.cpu().tolist())
            train_step += 1.0

        train_acc = accuracy_score(np.array(train_preds), np.array(train_labels))
        train_auc = roc_auc_score(np.array(train_labels), np.array(train_prob), average='macro', multi_class='ovo')
        print('epoch:{}\n train_loss:{}\n train_acc:{}\n train_auc:{}'.format(i, train_loss / train_step, train_acc, train_auc))

        dev_acc_, dev_auc_ = dev(dev_data=dev_data, model=model, args=args)

        if dev_acc <= dev_acc_:
            dev_acc = dev_acc_
            max_acc_index = i

            if args.save_model:
                save_model = 'model_{}.pth'.format(args.model_name)
                save_file = os.path.join(args.output_path, save_model)
                print(dev_auc_, save_file)
                torch.save(model.state_dict(), save_file)

        if i - max_acc_index > args.stop_num:
            break

    file = open('../result.txt', 'a')
    file.write('max_acc: {}, {}'.format(max_acc_index, dev_acc) + '\n')
    file.close()

    print('-----------------------------------------------------------------------------------------------------------')
    print('max_acc: {}, {}'.format(max_acc_index, dev_acc))
    print('-----------------------------------------------------------------------------------------------------------')


def dev(dev_data, model, args):
    model.eval()
    dev_len = len(dev_data)

    dev_preds = []
    dev_prob = []
    dev_labels = []

    with torch.no_grad():
        for m in trange(0, dev_len, args.batch_size):
            if m + args.batch_size < dev_len:
                dev_batch_data = dev_data[m: m+args.batch_size]
            else:
                dev_batch_data = dev_data[m: dev_len]
            text_input, text_mask, label = Batch(dev_batch_data, args)

            out = model(text_input, text_mask)

            pred = out.argmax(dim=-1).cpu().tolist()
            pred_prob = torch.softmax(out, -1).cpu().tolist()
            dev_preds.extend(pred)
            dev_prob.extend(pred_prob)
            dev_labels.extend(label.cpu().tolist())

    dev_acc = accuracy_score(np.array(dev_preds), np.array(dev_labels))
    dev_auc = roc_auc_score(np.array(dev_labels), np.array(dev_prob), average='macro', multi_class='ovo')

    print('dev_acc:{}\n dev_auc:{}'.format(dev_acc, dev_auc))
    return dev_acc, dev_auc


def main():
    args = init_arg_parser()
    setup_seed(args)

    train_path = os.path.join(args.data_path, 'THUCNews_train.txt')
    train_data = textReadOneEach(train_path)
    train_data = dataPro(train_data)
    dev_path = os.path.join(args.data_path, 'THUCNews_test.txt')
    dev_data = textReadOneEach(dev_path)
    dev_data = dataPro(dev_data)

    args.model_name = 'cnn'

    # 2060 cuda核心数： 1920， rtx 2080: 4608 2.44
    if args.model_name == 'bert':
        model = model_classification(args=args)
        lr = args.bert_lr
    elif args.model_name == 'lstm':
        # LSTM-CNN 128 256 0.5  (0.85 )
        # Attention-LSTM-CNN 512 768 0.5
        # CAM-LSTM-CNN 128 512 0.2
        # MTL-LSTM-CNN 128 512 0.5
        model = TextRCNN(embed_dim=64, hidden_dim=16, dropout=0.5, num_class=args.num_class, vob_size=5049)
        lr = args.lstm_lr
    elif args.model_name == 'cnn':
        # CNN 256, 64, 0.2
        # GAN-CNN 256, 128, 0.5
        # LW-GAN-CNN 768, 512, 0.2
        model = TextCNN(128, args.max_len, 16, 0.5, args.num_class, vocab_size=5049)
        lr = args.cnn_lr
    elif args.model_name == 'linear':
        model = DeepNet(128, args.num_class, 0.5)
        lr = args.cnn_lr

    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train(train_data, dev_data, model, optimizer, criterion, args)


def p(data, num=5):
    for i in range(num):
        print(data[i])


if __name__ == '__main__':
    args = init_arg_parser()
    main()
