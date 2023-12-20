# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchtext.vocab as vocab


class Config(object):
    """配置参数"""

    def __init__(self, task, project, dataset, embedding):
        self.task = task
        self.project = project
        self.model_name = 'TextRCNN'
        self.train_path = dataset + '/' + project + '/' + task + '/train.txt'  # 训练集
        self.dev_path = dataset + '/' + project + '/' + task + '/dev.txt'  # 验证集
        # self.test_path = dataset + '/' + project + '/' + task + '/test.txt'  # 测试集
        self.test_path = dataset + '/aquirement/data_pred/pred/php-src.txt'  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/' + project + '/' + task + '/class.txt', encoding='utf-8').readlines()]  # 类别名单
        self.vocab_path = dataset + '/' + project + '/' + task + '/vocab.pkl'  # 词表
        self.save_path = dataset + '/' + project + '/' + task + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = dataset + '/' + project + '/' + task + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/' + project + '/' + task + '/' + embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None  # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.dropout = 0.8  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 200  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 80  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-4  # 学习率
        self.embed = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 256  # lstm隐藏层
        self.num_layers = 2  # lstm层数
        self.code_embedding = 768


'''Recurrent Convolutional Neural Networks for Text Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.code_embedding, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed, config.num_classes)
        # self.codeBert = vocab.Vectors(
        #     name='D:\\PycharmProject\\experiment\\Chinese-Text-Classification-Pytorch-master\\THUCNews\\codeembedding\\embedding.txt',
        #     unk_init=torch.Tensor.normal_)
        # self.codebert_embedding = nn.Embedding.from_pretrained(self.codeBert.vectors, freeze=False)
        self.GraphCodeBert = vocab.Vectors(
            name='D:\PycharmProject\experiment\Chinese-Text-Classification-Pytorch-master\THUCNews\Fan\embedding_graphcodebert.txt',
            unk_init=torch.Tensor.normal_)
        self.graphCodeBERT_embedding = nn.Embedding.from_pretrained(self.GraphCodeBert.vectors, freeze=False)
        self.Code2Vec = vocab.Vectors(
            name='D:\PycharmProject\experiment\Chinese-Text-Classification-Pytorch-master\THUCNews\codeembedding\code2vec.txt',
            unk_init=torch.Tensor.normal_)
        self.Code2Vec_embedding = nn.Embedding.from_pretrained(self.Code2Vec.vectors, freeze=False)
        self.CodeSlice = vocab.Vectors(
            name='D:\PycharmProject\experiment\Chinese-Text-Classification-Pytorch-master\THUCNews\codeembedding\code_slicing_embedding.txt',
            unk_init=torch.Tensor.normal_)
        self.CodeSlice_embedding = nn.Embedding.from_pretrained(self.CodeSlice.vectors, freeze=False)

    def forward(self, x, code):
        x, _ = x
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        # code embedding
        code_emb = self.CodeSlice_embedding(code)
        # 将torch.Size([128, 1, 768])扩展为torch.Size([128, 80, 768])
        code_emb = torch.unsqueeze(code_emb, 1)
        compress = code_emb.expand(-1, 80, -1)

        out, _ = self.lstm(compress)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out
