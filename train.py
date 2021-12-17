import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from Config import *
from process_data import *
import torch.utils.data as Data
from torch.autograd import Variable
from tqdm import tqdm
from LSTM import *
from LSTM_Attention import *
from textcnn import TextCNN
import logging
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


logger = get_logger('log/cn_BiLSTM.log')

train_rights = []


def rightness(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


def get_data():
    print("Load Word2id...")
    word2id, tag2id = build_word2id(Config.train_path, Config.validation_path, Config.test_path)

    print("Load Data...")
    out = load_data(Config.train_path, Config.validation_path, Config.test_path, word2id, tag2id)

    print("Process Data...")
    x_train, y_train, x_validation, y_validation, x_test, y_test = process_data(out)

    train_data = Data.TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
    validation_data = Data.TensorDataset(torch.LongTensor(x_validation), torch.LongTensor(y_validation))
    test_data = Data.TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))

    train_data = Data.DataLoader(train_data, batch_size=Config.batch_size, shuffle=True, drop_last=True)
    validation_data = Data.DataLoader(validation_data, batch_size=Config.batch_size, shuffle=True, drop_last=True)
    test_data = Data.DataLoader(test_data, batch_size=Config.batch_size, shuffle=True, drop_last=True)

    vocab_size = len(word2id)
    tag_size = len(tag2id)

    return vocab_size, tag_size, train_data, validation_data, test_data


def train(vocab_size, tag_size, train_data, validation_data):
    model = LSTM_Attention(vocab_size, tag_size).cuda()
    criterion = nn.CrossEntropyLoss()
    optimzier = optim.Adam(model.parameters(), lr=Config.lr)
    train_acc_total = []
    logger.info('start training!')
    validation_acc_total = []
    validation_loss_total = []
    best_acc = 0
    best_model = None
    for epoch in range(Config.epoch):
        print("Epoch{}:".format(epoch + 1))
        train_loss = 0
        train_acc = 0
        model.train()
        true_label = []
        pred_label = []
        for i, data in tqdm(enumerate(train_data), total=len(train_data)):
            x, y = data
            x, y = Variable(x).cuda(), Variable(y).cuda()
            # forward
            out = model(x)
            loss = criterion(out, y)
            right = rightness(out, y)
            _, pre = torch.max(out, 1)
            num_acc = (pre == y).sum()
            train_acc += num_acc.data.item()
            true_label = np.append(true_label, y.tolist())
            pred_label = np.append(pred_label, pre.tolist())
            # backward
            optimzier.zero_grad()
            loss.backward()
            optimzier.step()
            train_rights.append(right)
            train_loss += loss.data.item()
            acc = accuracy_score(true_label, pred_label)
            train_acc_total.append(acc)
        logger.info('epoch [{}]: train loss is:{:.6f},train acc is:{:.6f}'
                    .format(epoch + 1, train_loss / (len(train_data) * Config.batch_size),
                            train_acc / (len(train_data) * Config.batch_size)))

        sk_acc, eval_acc, eval_loss = evaluate(model, validation_data, criterion)
        validation_acc_total.append(sk_acc)
        validation_loss_total.append(eval_loss / (len(validation_data) * Config.batch_size))
        if best_acc < (eval_acc / (len(validation_data) * Config.batch_size)):
            best_acc = eval_acc / (len(validation_data) * Config.batch_size)
            best_model = model
            logger.info('best acc is {:.6f},best model is changed'.format(best_acc))

    torch.save(best_model, 'model/BiLSTM.pth')
    logger.info('finish training!')
    np.save('save/BiLSTM/train_acc.npy', np.array(train_acc_total))
    np.save('save/BiLSTM/acc.npy', np.array(validation_acc_total))
    np.save('save/BiLSTM/loss.npy', np.array(validation_loss_total))


def evaluate(model, validation_data, criterion):
    model.eval()
    eval_loss = 0
    eval_acc = 0
    true_label = []
    pred_label = []

    for i, data in enumerate(validation_data):
        x, y = data
        with torch.no_grad():
            x = Variable(x).cuda()
            y = Variable(y).cuda()
        out = model(x)
        loss = criterion(out, y)
        eval_loss += loss.data.item()
        _, pre = torch.max(out, 1)
        true_label = np.append(true_label, y.tolist())
        pred_label = np.append(pred_label, pre.tolist())
        num_acc = (pre == y).sum()
        eval_acc += num_acc.data.item()

    logger.info('test loss is:{:.6f},test acc is:{:.6f}'
                .format(eval_loss / (len(validation_data) * Config.batch_size),
                        eval_acc / (len(validation_data) * Config.batch_size)))
    sk_acc = accuracy_score(true_label, pred_label)
    return sk_acc, eval_acc, eval_loss


if __name__ == '__main__':
    vocab_size, tag_size, train_data, validation_data, test_data = get_data()
    train(vocab_size, tag_size, train_data, validation_data)
