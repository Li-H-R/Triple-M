import numpy as np
from step1train_our_model import STtrans_C_FNN
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap
from scipy.stats import wasserstein_distance


def model_test(x, per):
    """
    :param x: 输入数据
    :return: 类别 + 残差
    """
    # 创建一个新的模型实例，用于加载参数
    net = STtrans_C_FNN()  # 请替换成你的模型定义
    # 加载已保存的模型参数
    save_path = f'Model_parameters/model_{per}%.pth'
    net.load_state_dict(torch.load(save_path))
    net.eval()
    with torch.no_grad():
        _, out, _ = net(x)

        x_re = out.view(x.shape)
    return x_re


def load_online_data():
    file_path_ab = 'test_dataset/test_abnormal.pt'
    data_ab = torch.load(file_path_ab)
    file_path = 'test_dataset/test_normal.pt'
    data_n = torch.load(file_path)

    return data_n, data_ab


def t_sne_plot(x_data, ab_data):
    """
    使用 t-SNE 降维并绘制正常数据和异常数据的散点图
    :param x_data: 原始数据，形状为 (n_samples, n_features)
    :param ab_data: 重构数据，形状为 (n_samples, n_features)
    :return: 绘制t-SNE图
    """
    x_data = x_data.reshape(len(x_data), -1)
    ab_data = ab_data.reshape(len(ab_data), -1)

    # 合并数据集
    combined_data = np.vstack([x_data, ab_data])

    # 使用t-SNE进行降维，目标维度为2，便于绘制
    tsne = TSNE(n_components=2, random_state=23)
    reduced_data = tsne.fit_transform(combined_data)

    # 分离降维后的数据
    normal_data_2d = reduced_data[:len(x_data)]
    anomaly_data_2d = reduced_data[len(x_data):]

    # 为了绘制等高线图，使用核密度估计来平滑数据
    kde_normal = gaussian_kde(normal_data_2d.T)
    kde_anomaly = gaussian_kde(anomaly_data_2d.T)

    # 创建网格
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    x_grid, y_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

    # 计算每个网格点的概率密度
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()])
    normal_density = kde_normal(grid_points).reshape(x_grid.shape)
    anomaly_density = kde_anomaly(grid_points).reshape(x_grid.shape)

    # 绘制等高线图
    plt.figure(figsize=(8, 6))
    # 自定义颜色：为正常数据使用蓝色
    normal_cmap = ListedColormap([55 / 255, 103 / 255, 149 / 255])  # 蓝色

    # 自定义颜色：为异常数据使用红色
    anomaly_cmap = ListedColormap([239 / 255, 138 / 255, 71 / 255])  # 红色
    # 绘制正常数据的等高线图
    plt.contour(x_grid, y_grid, normal_density, levels=10, cmap=normal_cmap, alpha=0.5)

    # 绘制异常数据的等高线图
    plt.contour(x_grid, y_grid, anomaly_density, levels=10, cmap=anomaly_cmap, alpha=0.5)

    # 绘制散点图（可选，标记正常数据和异常数据）
    plt.scatter(normal_data_2d[:, 0], normal_data_2d[:, 1], color=[55 / 255, 103 / 255, 149 / 255], label='Raw data',
                s=20)
    plt.scatter(anomaly_data_2d[:, 0], anomaly_data_2d[:, 1], color=[239 / 255, 138 / 255, 71 / 255],
                label='Reconstructing data', s=20)

    # 添加标题和标签
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(False)
    # 自动调整布局
    plt.tight_layout()
    plt.savefig(r'C:\Users\admin\Desktop\t_sne.jpg', format='jpg', dpi=800)
    plt.show()

    return normal_data_2d, anomaly_data_2d


def reshape_data(data):
    data_filtered = data.detach().numpy()
    return data_filtered.reshape(len(data_filtered), -1)


def distance_metrics(xx, yy):
    # 计算Wasserstein距离
    distance_d1 = wasserstein_distance(xx[:, 0], yy[:, 0])
    distance_d2 = wasserstein_distance(xx[:, 1], yy[:, 1])

    print("第一维Wasserstein Distance:", distance_d1)
    print("第二维Wasserstein Distance:", distance_d2)
    return (distance_d1 + distance_d2) / 2


if __name__ == '__main__':
    percent = 100        # 20, 40, 60, 80, 100
    raw_normal_data, raw_abnormal_data = load_online_data()
    normal_data = model_test(raw_normal_data, percent)
    abnormal_data = model_test(raw_abnormal_data, percent)

    # 正常数据
    raw_normal_data = reshape_data(raw_normal_data)
    normal_data = reshape_data(normal_data)

    # 异常数据
    raw_abnormal_data = reshape_data(raw_abnormal_data)
    abnormal_data = reshape_data(abnormal_data)

    # 绘图t-sne
    x, x_r = t_sne_plot(raw_normal_data, normal_data)
    x1 = distance_metrics(x, x_r)

    x, x_r = t_sne_plot(raw_abnormal_data, abnormal_data)
    x2 = distance_metrics(x, x_r)

    print('异常灵敏度:', np.sqrt(np.tanh(x2) * (np.tanh(-x1) + 1)))

