import torch
from torch.utils.data import Dataset
import numpy as np


class Dataset_Loader_processed(Dataset):
    def __init__(self, X_data, y_data, config):
        """
        数据集导入
        :param X_data:
        :param y_data:
        :param config:
        """
        self.X_data = X_data
        self.y_data = y_data
        self.config = config

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        feature = self.X_data[idx]
        label = self.y_data[idx]
        return {'vectors': torch.tensor(feature, dtype=torch.float32), 'labels': torch.tensor(label, dtype=torch.long)}


class Dataset_Loader_graph(Dataset):
    def __init__(self, data_X, data_y, config):
        self.data_X = data_X
        self.data_y = data_y
        self.config = config

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, idx):
        feature = self.data_X[idx]
        feature = np.array(feature)
        feature_reshaped = feature
        if self.config.aggreate == "mean":
            feature_reshaped = np.mean(feature, axis=0)
        elif self.config.aggreate == "sum":
            feature_reshaped = np.sum(feature, axis=0)
        else:
            dim1, dim2, dim3 = feature.shape
            feature_transposed = np.transpose(feature, (1, 0, 2))
            feature_reshaped = feature_transposed.reshape(dim2, dim1 * dim3)
        label = self.data_y[idx]
        vectors = np.zeros(shape=(self.config.max_node_len, self.config.embedding_size * self.config.channels))
        for i in range(min(feature_reshaped.shape[0], self.config.max_node_len)):
            vectors[i, :] = feature_reshaped[i, :]
        return {
            'vectors': torch.tensor(vectors, dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.long)
        }
