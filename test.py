import torch.nn as nn
import torch
from Config import *
from train_and_eval import get_data
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np

vocab_size, tag_size, train_data, validation_data, test_data = get_data()
model = torch.load('model/eng_LSTM.pth', map_location='cuda')
criterion = nn.CrossEntropyLoss()


def evaluate(model, test_data, criterion):
    model.eval()
    test_loss = 0
    test_acc = 0
    true_label = []
    pred_label = []

    for i, data in enumerate(test_data):
        x, y = data
        with torch.no_grad():
            x = Variable(x).cuda()
            y = Variable(y).cuda()
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


sk_loss, eval_acc, eval_loss, metrix = evaluate(model, test_data, criterion=criterion)
print(metrix)

