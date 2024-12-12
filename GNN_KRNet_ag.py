import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
from sklearn.metrics import accuracy_score
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import os

# 检查并设置 GPU 或 CPU 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def positional_encoding(seq_len, d_model):
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

    pos_enc = torch.zeros(seq_len, d_model)
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)

    return pos_enc


def mean_loss(xx, yy):
    uniq = torch.unique(yy)
    loss_mean = 0
    for label in uniq:
        # 表示向量
        rep_class = xx[yy == label] - torch.mean(xx[yy == label], dim=0)
        zreos_label = torch.zeros(rep_class.shape).detach()

        loss0 = criteria2(rep_class.contiguous().view(-1),
                          zreos_label.contiguous().view(-1))
        loss_mean += loss0
    return loss_mean


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


# 定义损失函数，这里引入了可训练参数A
class My_UW_Loss(nn.Module):
    """Uncertainty Weighting
       Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses
       for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and
       pattern recognition(pp. 7482-7491).
       """

    def __init__(self):
        super(My_UW_Loss, self).__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True))  # 可训练参数A
        self.b = nn.Parameter(torch.randn(1, requires_grad=True))  # 可训练参数A
        self.c = nn.Parameter(torch.randn(1, requires_grad=True))  # 可训练参数A
        self.log_vars = [self.a, self.c]

    def forward(self, losses):
        for i, log_var in enumerate(self.log_vars):
            losses[i] = (1 / 2) * (torch.exp(-log_var[0]) ** 2) * losses[i] + torch.log(torch.exp(log_var[0]) + 1)
        return sum(losses)


# 定义 KnowledgeRepresentationNet 模型
class Gnn_KRNet(torch.nn.Module):
    def __init__(self, input_dim=10, batch_size=256, encoder_dim=5):
        super(Gnn_KRNet, self).__init__()
        self.input_dim = input_dim
        self.conv1 = GCNConv(input_dim, input_dim * 2)
        self.conv2 = GCNConv(input_dim * 2, encoder_dim)
        self.vector_num = 6
        self.decoder_layer1 = nn.Linear(self.vector_num * encoder_dim, self.vector_num * input_dim)

        self.conv3_decoder = GCNConv(encoder_dim, input_dim * 2)
        self.conv4_decoder = GCNConv(input_dim * 2, input_dim)
        self.represent_layer = nn.Linear(self.vector_num * encoder_dim, 7)
        self.batch_size = batch_size

        self.positional_encoding = PositionalEncoding(self.vector_num * input_dim * 2, dropout=0)

        self.embedding_linear = nn.Linear(self.vector_num * encoder_dim, self.vector_num * encoder_dim)
        self.predictor = nn.Linear(self.vector_num * input_dim * 2, self.vector_num)
        self.ac = nn.Sigmoid()

    def forward(self, data):
        x, edge_index = data.x.to(torch.float32), data.edge_index
        # 原始数据位置编码
        po = positional_encoding(x.shape[1], 1).T
        x = x + po

        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.gelu(x)

        # 张量变换
        reshape_x = x.view(self.batch_size, self.vector_num * x.shape[1])

        # 进行一次线性变换竟然效果变好了！
        encoder_out = self.embedding_linear(reshape_x)
        # decoder2.0
        d_x = F.gelu(encoder_out)
        out = self.decoder_layer1(d_x)
        # 改变结构以方便分类器输入
        # 使用view不会生成新张量，即在原输出数据上进行改变
        x1 = self.represent_layer(encoder_out)

        return x1, encoder_out, out


if __name__ == '__main__':

    # 随机种子
    # 固定随机种子等操作
    seed_n = 40
    print('seed is ' + str(seed_n))
    g = torch.Generator()
    g.manual_seed(seed_n)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    os.environ['PYTHONHASHSEED'] = str(seed_n)  # 为了禁止hash随机化，使得实验可复现。

    # 超参数
    window_length = 11
    batch_size1 = 256
    num_epochs = 300

    # 导入数据
    loaded_data = torch.load('../data_process_graph/len11_have_graph.pt')

    # 模型
    model = Gnn_KRNet(input_dim=window_length, encoder_dim=7).to(device)

    # 损失函数
    criteria = nn.CrossEntropyLoss()
    criteria1 = nn.MSELoss()
    criteria2 = nn.MSELoss()
    loss_function = My_UW_Loss()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': loss_function.parameters()}], lr=3e-4)  # 将模型参数和损失函数参数一同传递给优化器

    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    # 定义编码器，词典大小为10，要把token编码成128维的向量
    Loss_get = []
    for epoch in range(num_epochs):
        total_loss = 0
        tt, tt2, tt3 = 0, 0, 0
        for batch in loaded_data:
            targets = torch.LongTensor(batch.y)
            output, output0, output1 = model(batch)

            # 目标数据
            tgt1 = batch.x.to(torch.float32)
            loss1 = criteria1(output1.contiguous().view(-1), tgt1.contiguous().view(-1))
            loss0 = mean_loss(output0, targets)
            loss = loss_function([loss1, loss0])
            total_loss += loss.item()

            tt += loss0.item()
            tt3 += loss1.item()
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('损失', total_loss)
        print('聚合损失', tt)
        print('重构损失', tt3)

        Loss_get.append([tt, tt2, tt3])

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss}')
            # 保存模型参数到文件
            torch.save(model.state_dict(), 'parameter_loss_ag.pth')

