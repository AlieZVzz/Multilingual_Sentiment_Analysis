import os.path
import torch.optim as optim
import Config
from process_data import *
import torch.utils.data as Data
from tqdm import tqdm
from SAmodel import *
import logging
from sklearn.metrics import accuracy_score
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


def get_data():
    print("Load Word2id...")
    word2id, tag2id = build_word2id(Config.train_path, Config.validation_path)

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


def evaluate(model, validation_data):
    model.eval()
    eval_loss, eval_acc = 0, 0
    for i, data in enumerate(validation_data):
        x, y = data
        with torch.no_grad():
            x = Variable(x).cuda()
            y = Variable(y).cuda()
        out = model(x)
        loss = criterion(out, y)
        eval_loss += loss.data.item()
        _, pred = torch.max(out, 1)
        num_acc = (pred == y).sum()
        eval_acc += num_acc.data.item()

    logger.info('validation loss is:{:.6f},validation acc is:{:.6f}'
                .format(eval_loss / (len(validation_data) * Config.batch_size),
                        eval_acc / (len(validation_data) * Config.batch_size)))
    return eval_acc, eval_loss


def train_and_eval(train_data, validation_data, criterion):
    best_acc = 0
    best_model = None
    logger.info('start training!')
    for epoch in range(Config.epoch):
        train_loss, train_acc = 0, 0
        true_label = []
        pred_label = []
        model.train()
        for i, data in tqdm(enumerate(train_data), total=len(train_data)):
            x, y = data
            x, y = Variable(x).cuda(), Variable(y).cuda()

            # forward
            out = model(x)

            loss = criterion(out, y)
            pred = out.argmax(axis=1)
            # _, pre = torch.max(out, 1)
            num_acc = (pred.type(y.dtype) == y).type(y.dtype).sum()
            train_acc += num_acc.data.item()
            true_label = np.append(true_label, y.tolist())
            pred_label = np.append(pred_label, pred.tolist())
            # backward
            optimzier.zero_grad()
            loss.backward()
            optimzier.step()
            train_loss += loss.data.item()
            acc = accuracy_score(true_label, pred_label)
            train_dict['train_acc'].append(acc)
        # logger
        logger.info('epoch [{}]: train loss is:{:.6f},train acc is:{:.6f}'
                    .format(epoch + 1, train_loss / (len(train_data) * Config.batch_size),
                            train_acc / (len(train_data) * Config.batch_size)))
        # 测试
        eval_acc, eval_loss = evaluate(model, validation_data)
        train_dict['validation_acc'].append(eval_acc / (len(validation_data) * Config.batch_size))
        train_dict['validation_loss'].append(eval_loss / (len(validation_data) * Config.batch_size))
        # 保存
        if best_acc < (eval_acc / (len(validation_data) * Config.batch_size)):
            best_acc = eval_acc / (len(validation_data) * Config.batch_size)
            best_model = model
            logger.info('best model is changed, best acc is {:.6f}'.format(best_acc))

    logger.info('finish training!')
    torch.save(best_model.state_dict(), 'model/BiLSTM.pth')
    np.save(os.path.join(save_path, 'train_acc.npy'), np.array(train_dict['train_acc']))
    np.save(os.path.join(save_path, 'val_acc.npy'), np.array(train_dict['validation_acc']))
    np.save(os.path.join(save_path, 'val_acc.npy'), np.array(train_dict['validation_loss']))


if __name__ == '__main__':
    vocab_size, tag_size, train_data, validation_data, test_data = get_data()
    train_dict = {'train_acc': [], 'train_loss': [], 'validation_acc': [], 'validation_loss': []}
    logger = get_logger('log/cn_BiLSTM.log')
    save_path = 'save/BiLSTM'
    model = LSTM_Attention(vocab_size, tag_size).cuda()
    criterion = nn.CrossEntropyLoss()
    optimzier = optim.Adam(model.parameters(), lr=Config.lr)
    train_and_eval(train_data, validation_data, criterion)
