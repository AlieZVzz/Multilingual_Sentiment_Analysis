import torch
import torch.nn as nn
from Config import *
from torch.autograd import Variable
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, vocab_size, tag_size):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, Config.embedding_dim)
        self.lstm = nn.LSTM(input_size=Config.embedding_dim, hidden_size=Config.hidden_dim,
                            bidirectional=Config.bidirectional, num_layers=Config.layer_size, dropout=0.5)
        self.classify = nn.Linear(Config.hidden_dim * Config.num_direction, tag_size)

    def forward(self, input):
        # seq_len = 60  句子长度

        input = self.embed(input)  # [64, 60, 100]  [batch_size, seq_len, hidden_dim]
        input = input.permute(1, 0, 2)

        out, _ = self.lstm(input)  # [60, 64, 100]  [seq_len, batch_size, hidden_dim]

        out = out[-1, :, :]  # [64, 100]
        out = out.view(-1, Config.hidden_dim * Config.num_direction)  # [64, 100]  [batch_size, seq_len]
        out = self.classify(out)  # [64, 2]

        return out


class LSTM_Attention(nn.Module):
    def __init__(self, vocab_size, tag_size):
        super(LSTM_Attention, self).__init__()
        self.embed = nn.Embedding(vocab_size, Config.embedding_dim)
        self.lstm = nn.LSTM(input_size=Config.embedding_dim, hidden_size=Config.hidden_dim,
                            bidirectional=Config.bidirectional)
        self.classify = nn.Linear(
            Config.hidden_dim * Config.num_direction, tag_size)

        self.w_omega = Variable(torch.zeros(
            Config.hidden_dim * Config.num_direction, Config.attention_size)).cuda()
        self.u_omega = Variable(torch.zeros(Config.attention_size)).cuda()

    def attention_net(self, out):
        # out:[seq_len, batch_size, hidden_dim * num_direction]

        output_reshape = torch.Tensor.reshape(out, [-1,
                                                    Config.hidden_dim * Config.num_direction])  # [seq_len * batch_size, hidden_dim * num_direction]

        # [seq_len * batch_size, attention_size]
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))

        attn_hidden_layer = torch.mm(attn_tanh,
                                     torch.Tensor.reshape(self.u_omega, [-1, 1]))  # [seq_len * batch_size, 1]

        exps = torch.Tensor.reshape(torch.exp(
            attn_hidden_layer), [-1, Config.sequence_length])  # [batch_size, seq_len]

        # [batch_size, seq_len]
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])

        alphas_reshape = torch.Tensor.reshape(
            alphas, [-1, Config.sequence_length, 1])  # [batch_size, seq_len, 1]

        # [batch_size, seq_len, hidden_dim * num_direction]
        state = out.permute(1, 0, 2)

        # [batch_size, hidden_dim * num_direction]
        attn_output = torch.sum(state * alphas_reshape, 1)

        return attn_output

    def forward(self, input):
        # seq_len = 60  句子长度

        # [64, 60, 100]  [batch_size, seq_len, hidden_dim]
        input = self.embed(input)
        input = input.permute(1, 0, 2)

        # [60, 64, 100]  [seq_len, batch_size, hidden_dim * num_direction]
        out, _ = self.lstm(input)

        attn_output = self.attention_net(out)

        out = self.classify(attn_output)  # [64, 2]

        return out


class TextCNN(nn.Module):
    def __init__(self, vocab_size, tag_size):
        super(TextCNN, self).__init__()

        label_num = tag_size  # 标签的个数
        filter_num = Config.filter_num  # 卷积核的个数
        filter_sizes = [int(fsz) for fsz in Config.filter_sizes.split(',')]

        vocab_size = vocab_size
        embedding_dim = Config.embedding_dim

        self.embedding = nn.Embedding(vocab_size, Config.embedding_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, embedding_dim)) for fsz in filter_sizes])
        self.dropout = nn.Dropout(Config.dropout)
        self.linear = nn.Linear(len(filter_sizes) * filter_num, label_num)

    def forward(self, x):
        # 输入x的维度为(batch_size, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大=长度
        x = self.embedding(x)  # 经过embedding,x的维度为(batch_size, max_len, embedding_dim)

        # 经过view函数x的维度变为(batch_size, input_chanel=1, w=max_len, h=embedding_dim)
        x = x.unsqueeze(1)

        # 经过卷积运算,x中每个运算结果维度为(batch_size, out_chanel, w, h=1)
        x = [F.relu(conv(x)) for conv in self.convs]

        # 经过最大池化层,维度变为(batch_size, out_chanel, w=1, h=1)
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]

        # 将不同卷积核运算结果维度（batch，out_chanel,w,h=1）展平为（batch, outchanel*w*h）
        x = [x_item.view(x_item.size(0), -1) for x_item in x]

        # 将不同卷积核提取的特征组合起来,维度变为(batch, sum:outchanel*w*h)
        x = torch.cat(x, 1)

        # dropout层
        x = self.dropout(x)
        logits = self.linear(x)
        return logits
