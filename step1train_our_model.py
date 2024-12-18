import torch
import pickle
import torch.nn as nn
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F


def data_used(per):
    save_path = f'train_dataset/KIL_DataSet_{per}%.pt'
    data_set = torch.load(save_path)

    return data_set


# 位置编码
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


class STtrans_C_FNN(nn.Module):

    def __init__(self, input_dim=13, d_model=16, window_l=11, encoder_l=128):
        super(STtrans_C_FNN, self).__init__()

        # 原始数据维度为13，扩展编码为16 便于多投注意力机制
        self.embedding_encoder_T = nn.Linear(input_dim, d_model)
        self.embedding_encoder_S = nn.Linear(window_l, d_model)
        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(d_model, dropout=0)
        self.encoder_time = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,  # 输入数据的特征维度
                nhead=8,  # 注意力头数
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=1,
        )
        self.encoder_space = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,  # 输入数据的特征维度
                nhead=8,  # 注意力头数
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=1,
        )

        self.space_linear = nn.Linear(input_dim, d_model)
        self.layer_fusion = nn.Linear((input_dim + window_l) * d_model, encoder_l)

        # 解码器
        self.layers_decoder = nn.Sequential(
            nn.Linear(encoder_l, 160),
            nn.GELU(),
            nn.Linear(160, 160),
            nn.GELU(),
            nn.Linear(160, input_dim * window_l)
        )
        # 仿射层
        self.re_linear = nn.Linear(6*7, encoder_l)

    def forward(self, src, k_i=None):
        # 对src进行编码
        src_time = self.embedding_encoder_T(src)
        src_space = src.permute(0, 2, 1)
        src_space = self.embedding_encoder_S(src_space)

        # 给src和tgt的token增加位置信息
        src_time = self.positional_encoding(src_time)

        # 将准备好的数据送给transformer
        memory_out_time = self.encoder_time(src_time)
        # 不用转置，使得存储连续，方便训练
        memory_out_space = self.encoder_space(src_space)

        memory_out = torch.cat((memory_out_time, memory_out_space), dim=1)
        reshape_x = memory_out.view(-1, memory_out.shape[1] * memory_out.shape[2])
        memory_out = self.layer_fusion(reshape_x)
        x_re = self.layers_decoder(memory_out)
        if k_i is not None:
            text_out = self.re_linear(k_i)
        else:
            text_out = None  # or provide a default value if needed
        return memory_out, x_re, text_out

def train_clip(image, text_features, y):
    # 假设 image_features 和 text_features 都是已经处理过的特征，image 即为 image_features
    image_features = image

    # 归一化 text_features 和 image_features（按行归一化）
    text_features_normalized = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features_normalized = image_features / image_features.norm(dim=-1, keepdim=True)

    # 计算余弦相似度
    logits = text_features_normalized @ image_features_normalized.T  # 点积得到相似度

    # 目标标签
    unique_y, inverse_indices = torch.unique(y, return_inverse=True)
    labels_clip = torch.zeros(logits.shape)

    for y_label in unique_y:
        mask = (y == y_label)
        target_y = torch.zeros(y.shape)
        target_y[mask] = 1
        labels_clip[y == y_label] = target_y

    logits = torch.softmax(logits, dim=-1)  # softmax 激活
    target1 = torch.softmax(labels_clip, dim=-1)  # softmax 激活目标标签

    return criteria1(logits, target1), logits, target1


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
        self.log_vars = [self.a, self.b]

    def forward(self, losses):
        for i, log_var in enumerate(self.log_vars):
            losses[i] = (1 / 2) * (torch.exp(-log_var[0]) ** 2) * losses[i] + torch.log(torch.exp(log_var[0]) + 1)
        return sum(losses)


if __name__ == '__main__':
    percent = 100  # 20, 40, 60, 80
    # 随机种子
    # 固定随机种子等操作 30, 32
    seed_n = 30
    print('seed is ' + str(seed_n))
    g = torch.Generator()
    g.manual_seed(seed_n)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    os.environ['PYTHONHASHSEED'] = str(seed_n)  # 为了禁止hash随机化，使得实验可复现。

    # 输入数据
    loaded_data = data_used(percent)
    # 超参数
    input_dim = 13
    num_epochs = 300
    batch_size = 64
    zero_dim = torch.zeros(batch_size, 1, input_dim) + 2
    learning_rate = 3e-4

    # 模型
    model = STtrans_C_FNN()
    # 开启dropout
    model.train()
    loss_function = My_UW_Loss()
    # 优化器
    optimizer_encoder = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_function = torch.optim.Adam(loss_function.parameters(), lr=learning_rate)
    # 损失函数
    criteria = nn.MSELoss()
    criteria1 = nn.MSELoss(reduction='sum')
    # 训练
    for epoch in range(num_epochs):
        total_loss = 0
        total_loss1, total_loss2 = 0, 0

        for batch in loaded_data:
            inputs, y_label, targets, know_inputs = batch
            inputs = inputs.to(torch.float32)
            targets = targets.to(torch.float32)
            know_inputs = know_inputs.to(torch.float32)

            m_out, outs, re_o = model(inputs, know_inputs)
            # 清空梯度
            optimizer_encoder.zero_grad()
            optimizer_function.zero_grad()

            # CLIP损失
            loss_clip, a, b = train_clip(m_out, re_o, y_label)
            # 重构损失
            loss_re = criteria(outs.contiguous().view(-1), targets.contiguous().view(-1))
            loss = loss_function([loss_clip, loss_re])
            # 计算梯度
            loss.backward()

            # 更新参数
            optimizer_encoder.step()
            optimizer_function.step()

            total_loss += loss.item()
            total_loss1 += loss_re.item()
            total_loss2 += loss_clip.item()
        print('总损失', total_loss)
        print('重构损失', total_loss1)
        print('匹配损失', total_loss2)

        # if (epoch + 1) % 2 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss}')
        # 保存模型参数
        save_path = f'Model_parameters/model_{percent}%.pth'
        torch.save(model.state_dict(), save_path)


