import numpy as np


data = np.random.rand(283,8,256)
print(data.shape)
from sklearn.decomposition import PCA

pca_results = []
for t in range(8):
    pca = PCA(n_components=t+1)  #
    time_slice = data[:, t, :]
    principal_components = pca.fit_transform(time_slice)
    pca_results.append(principal_components)

final_pca_results = np.hstack(pca_results)
# print(final_pca_results)

close_prices = final_pca_results

correlation_matrix = np.corrcoef(close_prices)  #

threshold = 0.5
spatial_weights = np.where(correlation_matrix > threshold, correlation_matrix, 0)

row_sums = spatial_weights.sum(axis=1)
spatial_weights = np.divide(spatial_weights, row_sums[:, np.newaxis], where=row_sums[:, np.newaxis] != 0)
print(spatial_weights.shape)


import libpysal
from esda.moran import Moran_BV

w = libpysal.weights.full2W(spatial_weights)  # spatial_weights
moran_index_X1_X2 = Moran_BV(principal_components[:, 0], principal_components[:, 1], w)
print(moran_index_X1_X2.I)  #

import numpy as np
import libpysal
from esda.moran import Moran_BV


import numpy as np
import libpysal
from esda.moran import Moran
from sklearn.decomposition import PCA

pca_results = []
for t in range(8):
    pca = PCA(n_components=t+1)
    time_slice = data[:, t, :]  #
    principal_components = pca.fit_transform(time_slice)
    pca_results.append(principal_components)

final_pca_results = np.hstack(pca_results)


# normalized_data = data.detach().cpu().numpy()
final_pca_results = np.corrcoef(final_pca_results)
print(final_pca_results)
w = libpysal.weights.full2W(final_pca_results)
print(w)

moran_indices = np.zeros((283, 10))
reshaped_data = data.reshape(data.shape[0], -1)


pca = PCA(n_components=10)
features = pca.fit_transform(reshaped_data)

for feature_idx in range(10):
    feature_data = features[:, feature_idx]
    moran_i = Moran(feature_data, w)
    moran_indices[:, feature_idx] = moran_i.I

correlation_matrix = np.mean(moran_indices, axis=1)
correlation_matrix = np.outer(correlation_matrix, correlation_matrix)
print(correlation_matrix)




import numpy as np
import libpysal
from esda.moran import Moran_BV
from sklearn.preprocessing import scale


features = np.random.rand(283, 10)
spatial_weights = np.random.rand(283, 283)


W = libpysal.weights.full2W(spatial_weights)
W.transform = 'R'


features = scale(features)

import numpy as np
import pandas as pd

features = np.random.rand(283,10)
weights = np.random.rand(283,283)

normalized_features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

cross_moran_matrix = np.zeros((283, 283))


for i in range(283):
    for j in range(283):
        sum_weighted_products = 0
        for k in range(283):
            if k != i and k != j:
                weighted_product = weights[i, k] * normalized_features[k, :] * normalized_features[j, :]
                sum_weighted_products += weighted_product
        #
        cross_moran_matrix[i, j] = sum_weighted_products.mean()


print(cross_moran_matrix)

