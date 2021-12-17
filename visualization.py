import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

attention_acc = np.load('save/eng_LSTM_attention/acc.npy')
attention_loss = np.load('save/eng_LSTM_attention/loss.npy')
eng_attention_loss = np.load('save/eng_LSTM_attention/loss.npy')

eng_CNN_acc = np.load('save/eng_CNN/acc.npy')
eng_CNN_loss = np.load('save/eng_CNN/loss.npy')
loss = np.load('save/LSTM/loss.npy')
acc = np.load('save/LSTM/acc.npy')
plt.plot(attention_acc)
plt.plot(acc)
plt.plot(eng_CNN_acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(labels=['Attention LSTM', 'LSTM', 'Texteng_CNN'], loc='lower right')
plt.show()
plt.plot(attention_loss)
plt.plot(loss)
plt.plot(eng_CNN_loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(labels=['Attention LSTM', 'LSTM', 'Texteng_CNN'], loc='lower right')
plt.show()
