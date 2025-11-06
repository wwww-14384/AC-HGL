import numpy as np
import pandas as pd
import copy
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.optim as optim
from scipy.stats import zscore
from scipy.stats import spearmanr

import numpy as np
from scipy.stats import iqr


# def visualize_explanation(edge_mask: np.ndarray, threshold=None, node_labels=None):
#     """可视化边重要性矩阵
#
#     Args:
#         edge_mask (np.ndarray):
#         threshold (float):
#         node_labels (list):
#     """
#     if threshold is None:
#         threshold = np.percentile(edge_mask, 90)
#
#     plt.figure(figsize=(12, 10))
#     plt.imshow(edge_mask > threshold, cmap='viridis')
#
#     if node_labels:
#         plt.xticks(range(len(node_labels)), node_labels, rotation=90)
#         plt.yticks(range(len(node_labels)), node_labels)
#
#     plt.title("Top 10% Important Connections")
#     plt.colorbar()
#     plt.savefig('stock_relations.png', bbox_inches='tight')
#     plt.close()


def robust_zscore_norm_multidimensional(data, axis=2, clip_outlier=True, threshold=3.0):

    data = data.detach().cpu().numpy()
    median = np.median(data, axis=axis, keepdims=True)
    iqr_value = iqr(data, axis=axis, keepdims=True)


    iqr_value[iqr_value == 0] = 1


    normalized_data = (data - median) / iqr_value


    if clip_outlier:
        normalized_data = np.clip(normalized_data, -threshold, threshold)

    return normalized_data

def calc_ic(pred, label):
    # pred = pred.flatten()
    # label = label.flatten()
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric
def spearman(data):
    data = data.t()
    normalized_data = zscore(data.detach().cpu().numpy(), axis=1)


    corr, _ = spearmanr(normalized_data.T, axis=1)

    return corr


import libpysal

from esda.moran import Moran_BV
from sklearn.preprocessing import scale
import numpy as np
from sklearn.decomposition import PCA

def calculate_correlation_all_features(data):
    # data = data.t()
    # normalized_data = zscore(data.detach().cpu().numpy(),axis=1)
    # normalized_data = zscore(normalized_data,axis=0)
    normalized_data = data.detach().cpu().numpy()
    correlation_matrix = np.corrcoef(normalized_data)
    # print(correlation_matrix.shape)
    return correlation_matrix
def calculate_moran(data):
    # data = data.t()
    # normalized_data = zscore(data.detach().cpu().numpy(),axis=1)
    # normalized_data = zscore(normalized_data,axis=0)
    data = data.detach().cpu().numpy()

    # pca_results = []
    # for t in range(8):
    #     pca = PCA(n_components=(t + 1)
    #     time_slice = data[:, t, :]
    #     principal_components = pca.fit_transform(time_slice)
    #     pca_results.append(principal_components)
    #
    # features = np.hstack(pca_results)

    pca_results = []
    for t in range(8):
        pca = PCA(n_components=2)
        time_slice = data[:, t, :]
        principal_components = pca.fit_transform(time_slice)
        pca_results.append(principal_components)
        # print(pca_results)
    features = np.hstack(pca_results)

    # features = data.reshape(data.shape[0],-1)
    n = 16

    weights_matrix = np.zeros((n, n))


    weights_matrix = np.full((n, n), 1)
    np.fill_diagonal(weights_matrix, 0)  # 将对角线设为0


    w = libpysal.weights.full2W(weights_matrix)
    w.transform = 'R'  # 行标准化

    # features = scale(features)


    cross_moran_matrix = np.zeros((data.shape[0], data.shape[0]))
    cross_p_matrix = np.zeros((data.shape[0], data.shape[0]))
    cross_z_matrix = np.zeros((data.shape[0], data.shape[0]))

    for i in range(data.shape[0]):
        for j in range(i + 1, data.shape[0]):
            moran_bv = Moran_BV([features[i]], [features[j]], w, permutations=99)

            cross_moran_matrix[i, j] = moran_bv.I
            cross_moran_matrix[j, i] = moran_bv.I
            cross_p_matrix[i, j] = moran_bv.p_sim
            cross_p_matrix[j, i] = moran_bv.p_sim
            cross_z_matrix[i, j] = moran_bv.z_sim
            cross_z_matrix[j, i] = moran_bv.z_sim
    # print(cross_moran_matrix)
    # print(cross_p_matrix)
    return cross_moran_matrix,cross_p_matrix,cross_z_matrix













#




def calculate_correlation_all_features(data):

    # print(data.shape)
    data = data.detach().cpu().numpy()


    mean_data = np.mean(data, axis=1, keepdims=True)


    data_centered = data - mean_data


    covariance_matrix = np.dot(data_centered, data_centered.T)


    variance_data = np.sum(data_centered ** 2, axis=1)


    denominator = np.sqrt(np.outer(variance_data, variance_data))
    correlation_matrix = covariance_matrix / denominator

    return correlation_matrix


# def calculate_correlation_all_features(data):
#     # data = data.t()
#     # normalized_data = zscore(data.detach().cpu().numpy(),axis=1)
#     # normalized_data = zscore(normalized_data,axis=0)
#     normalized_data = data.detach().cpu().numpy()
#     correlation_matrix = np.corrcoef(normalized_data)
#     # print(correlation_matrix.shape)
#     return correlation_matrix
class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch


        self.daily_count = pd.Series(index=self.data_source.get_index(),dtype=pd.StringDtype()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


class SequenceModel():
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path = 'model/', save_prefix= ''):
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        self.fitted = False

        self.model = None
        self.train_optimizer = None

        self.save_path = save_path
        self.save_prefix = save_prefix


    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        loss = (pred[mask]-label[mask])**2
        return torch.mean(loss)

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []
        i = 1
        for data in data_loader:
            # print(data.size())
            #data_loader:2979(2008-2020
            # print("data:",data.size(),data)
            # data: torch.Size([1, 722, 8, 222])
            # data: torch.Size([1, 734, 8, 222])
            # data: torch.Size([1, 653, 8, 222])
            data = torch.squeeze(data, dim=0)
            # print("data",data.size())
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158 factors + 63 market information + 1 label           
            '''
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)


            pred = self.model(feature.float(),label.float(),require_exp=True)

            loss = self.loss_fn(pred, label)






            losses.append(loss.item())
            if i%120==0:
                print(f'date {i}',loss.item())
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()
            # print("i：",i)
            # # i： 2979
            i+=1


            # if i == 2977:
            #     print(222222222222222222222222)
            # #     torch.save(pos_stream, 'pos_stream_2977.pt')
            # #     torch.save(pos_attn, 'pos_attn_2977.pt')
            # #     torch.save(pos_cor_matrix, 'pos_cor_matrix_2977.pt')
            # #     torch.save(neg_stream, 'neg_stream_2977.pt')
            # #     torch.save(neg_attn, 'neg_attn_2977.pt')
            # #     torch.save(neg_cor_matrix, 'neg_cor_matrix_2977.pt')
            # #     torch.save(t_src, 't_src_2977.pt')
            # #     torch.save(all_src, 'all_src_2977.pt')
            # #     torch.save(hete_attn, 'hete_attn_2977.pt')
            #     torch.save(t_matrix, 't_matrix_2977.pt')
            # #     # torch.save(src_pre, 'src_pre.pt')
            # #     # torch.save(src_pos,'src_pos.pt')
            # #     # torch.save(src_neg,'src_neg.pt')
            # if i == 2979:
            #     print(222222222222222222222222)
            # #     torch.save(pos_stream, 'pos_stream_2979.pt')
            # #     torch.save(pos_attn, 'pos_attn_2979.pt')
            # #     torch.save(pos_cor_matrix, 'pos_cor_matrix_2979.pt')
            # #     torch.save(neg_stream, 'neg_stream_2979.pt')
            # #     torch.save(neg_attn, 'neg_attn_2979.pt')
            # #     torch.save(neg_cor_matrix, 'neg_cor_matrix_2979.pt')
            # #     torch.save(t_src, 't_src_2979.pt')
            # #     torch.save(all_src, 'all_src_2979.pt')
            # #     torch.save(hete_attn, 'hete_attn_2979.pt')
            #     torch.save(t_matrix, 't_matrix_2979.pt')
            # #     # torch.save(src_pre, 'src_pre.pt')
            # #     # torch.save(src_pos,'src_pos.pt')
            # #     # torch.save(src_neg,'src_neg.pt')
            # if i == 2978:
            #     print(33333333333333333333333)
            # #     torch.save(pos_stream, 'pos_stream_2978.pt')
            # #     torch.save(pos_attn, 'pos_attn_2978.pt')
            # #     torch.save(pos_cor_matrix, 'pos_cor_matrix_2978.pt')
            # #     torch.save(neg_stream, 'neg_stream_2978.pt')
            # #     torch.save(neg_attn, 'neg_attn_2978.pt')
            # #     torch.save(neg_cor_matrix, 'neg_cor_matrix_2978.pt')
            # #     torch.save(t_src, 't_src_2978.pt')
            # #     torch.save(all_src, 'all_src_2978.pt')
            # #     torch.save(hete_attn, 'hete_attn_2978.pt')
            #     torch.save(t_matrix, 't_matrix_2978.pt')
            # #         # torch.save(src_pre, 'src_pre.pt')
            # #         # torch.save(src_pos,'src_pos.pt')
            # #         # torch.save(src_neg,'src_neg.pt')
        print('day:', i - 1)
        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
            self.model = self.model

            pred = self.model(feature.float(),label.float())

            loss = self.loss_fn(pred, label)


        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        # print('data:',data)
        # print('sampler:',sampler)
        data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
        # print('data_loader:',data_loader)
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = True

    def fit(self, dl_train, dl_valid):
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        valid_loader = self._init_data_loader(dl_valid, shuffle=False, drop_last=True)
        self.fitted = True
        best_param = None

        for step in range(self.n_epochs):

            train_loss = self.train_epoch(train_loader)
            val_loss = self.test_epoch(valid_loader)
            # writer.add_scalar('Loss/Train', train_loss, step)
            # writer.add_scalar('Loss/Val', val_loss, step)
            # tensorboard - -logdir = data

            print("Epoch %d, train_loss %.6f, valid_loss %.6f " % (step, train_loss, val_loss))
            best_param = copy.deepcopy(self.model.state_dict())

            if train_loss <= self.train_stop_loss_thred:
                break
        torch.save(best_param, f'{self.save_path}{self.save_prefix}master_{self.seed}.pkl')



    def predict(self, dl_test):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)

        # print('test_loader:',test_loader)
        preds = []
        ic = []
        ric = []
        labels = []
        all_preds = []
        all_labels = []
        self.model.eval()
        for data in test_loader:
            # print('data:',data)
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
            with torch.no_grad():
                pred = self.model(feature.float(),label.float(),require_exp=True)

                # edge_mask = self.model.edge_importance.cpu().numpy()
                # top_threshold = np.percentile(edge_mask, 90)  # 取前10%的边

                # 3. 绘制关联网络
                # plt.figure(figsize=(12, 10))
                # visualize_explanation(
                #     edge_mask,
                #     threshold=top_threshold,
                #     # node_labels=stock_names  # 可传入股票名称列表作为标签
                # )
                # plt.savefig('stock_relations22.png')

                pred = pred.detach().cpu().numpy()
            # all_preds.extend(pred.ravel().tolist())
            # all_labels.extend(label.detach().numpy().tolist())

            preds.append(pred.ravel())
            # labels.append(label.detach().numpy())
            daily_ic, daily_ric = calc_ic(pred, label.detach().cpu().numpy())

            ic.append(daily_ic)
            ric.append(daily_ric)


        predictions = pd.Series(np.concatenate(preds), index=dl_test.get_index())
        print('****beta****:', self.beta)
        print('t_nhead', self.t_nhead, 's_nhead:', self.s_nhead)
        print('n_epochs:', self.n_epochs)
        print('cor:',self.cor)

        metrics = {
            'IC': np.mean(ic),
            'ICIR': np.mean(ic) / np.std(ic),
            'RIC': np.mean(ric),
            'RICIR': np.mean(ic) / np.std(ric)
        }

        return predictions, metrics
import pickle
# import pandas as pd
# f = open('/home/MASTER/data/csi300/csi300_dl_train.pkl','rb')
# data = pickle.load(f)
# print(data)
# import qlib.data.dataset.TSDataSampler


# # pd.set_option('display.width',None)
# # pd.set_option('display.max_rows',None)
# # pd.set_option('display.max_colwidth',None)
# print(data)
# inf=str(data)
# ft = open('/home/MASTER/data/csi300/csi300_dl_train.csv', 'w')
# ft.write(inf)

# f = open('/home/MASTER/data/csi300/csi300_dl_train.pkl','rb')
# data = pickle.load(f)
# print(data)
# daily_count = pd.Series(index=data.get_index(), dtype=pd.StringDtype()).groupby("datetime").size().values
# daily_index = np.roll(np.cumsum(daily_count), 1)  # calculate begin index of each batch
# daily_index[0] = 0
# print(daily_count[290])
# print(daily_index[1])
# data=pd.Series(index=data.get_index(), dtype=pd.StringDtype()).groupby("datetime")
# print(data.size())