import numpy as np


def get_knn_graph(data, K_num):
    # each row of data is a sample，每一行数据都是一个样本
    x_norm = np.reshape(np.sum(np.square(data), 1), [-1, 1])  # column vector 列向量
    x_norm2 = np.reshape(np.sum(np.square(data), 1), [1, -1])  # row vector 列向量
    dists = x_norm - 2 * np.matmul(data, np.transpose(data)) + x_norm2
    num_sample = data.shape[0]
    graph = np.zeros((num_sample, num_sample), dtype=np.int)

    for i in range(num_sample):
        distance = dists[i, :]
        small_index = np.argsort(distance)
        graph[i, small_index[0:K_num]] = 1

    graph = graph - np.diag(np.diag(graph))
    resultgraph = np.maximum(graph, np.transpose(graph))

    return resultgraph
