import torch.nn as nn
import torch
from Config import *
from train_and_eval import get_data
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from SAmodel import TextCNN
import numpy as np


def evaluate(model, test_data, criterion):
    model.eval()
    test_loss = 0
    test_acc = 0
    true_label = []
    pred_label = []

    for i, data in enumerate(test_data):
        x, y = data
        with torch.no_grad():
            x = Variable(x)
            y = Variable(y)
        out = model(x)
        loss = criterion(out, y)
        test_loss += loss.data.item()
        _, pre = torch.max(out, 1)
        true_label = np.append(true_label, y.tolist())
        pred_label = np.append(pred_label, pre.tolist())
        num_acc = (pre == y).sum()
        test_acc += num_acc.data.item()

    print('test loss is:{:.6f},test acc is:{:.6f}'
          .format(test_loss / (len(test_data) * Config.batch_size),
                  test_acc / (len(test_data) * Config.batch_size)))
    sk_acc = accuracy_score(true_label, pred_label)
    confusion_m = confusion_matrix(true_label, pred_label)
    return sk_acc, test_acc, test_loss, confusion_m


vocab_size, tag_size, train_data, validation_data, test_data = get_data()
state_dict = torch.load('model/TextCNN.pth', map_location='cpu')
model = TextCNN(vocab_size, tag_size)
model.load_state_dict(state_dict=state_dict)
criterion = nn.CrossEntropyLoss()
sk_loss, eval_acc, eval_loss, matrix = evaluate(model, test_data, criterion=criterion)
print(matrix)
