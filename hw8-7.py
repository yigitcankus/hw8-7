import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from scipy.stats import jarque_bera

import warnings
warnings.filterwarnings('ignore')

# df = pd.read_csv("final_dataa.csv")
#
# df.drop(df.columns[[0, 2, 3, 15, 17, 18]], axis=1, inplace=True)
#
#
# df['zindexvalue'] = df['zindexvalue'].str.replace(',', '')
# df["zindexvalue"]=df["zindexvalue"].astype(np.int64)
#
#
# #since the most correlated value is finishedsqft I made a new feature called price_per_sqft
# df['price_per_sqft'] = df['lastsoldprice']/df['finishedsqft']
# corr_matrix = df.corr()
#
# #but this new feature didnt make a big impact
#
# freq = df.groupby('neighborhood').size()
# mean = df.groupby('neighborhood').mean()['price_per_sqft']
# cluster = pd.concat([freq, mean], axis=1)
# cluster['neighborhood'] = cluster.index
# cluster.columns = ['freq', 'price_per_sqft','neighborhood']
# #minik bir veri kümesi oluşturduk. Hangi neighborhoodda kaç ev olduğunu ve bunların price per sqft(square footage) fiyatlarını bulduk
#
#
# cluster1 = cluster[cluster.price_per_sqft < 756]
#
# cluster_temp = cluster[cluster.price_per_sqft >= 756]
# cluster2 = cluster_temp[cluster_temp.freq <123]
#
# cluster3 = cluster_temp[cluster_temp.freq >=123]
#
# def get_group(x):
#     if x in cluster1.index:
#         return 'low_price'
#     elif x in cluster2.index:
#         return 'high_price_low_freq'
#     else:
#         return 'high_price_high_freq'
# df['group'] = df.neighborhood.apply(get_group)
# # I categorized the new feature cluster into 3.
#
# n = pd.get_dummies(df.group)
# df = pd.concat([df, n], axis=1)
# m = pd.get_dummies(df.usecode)
# df = pd.concat([df, m], axis=1)
# drops = ['group', 'usecode']
# df.drop(drops, inplace=True, axis=1)
# # I got the dummies of the groups and usecode columns
#
#
# def is_new(row):
#     if row["yearbuilt"] > 2005:
#         return 1
#     else:
#         return 0
#
# df["is_new"] = df.apply(is_new, axis=1)
# # I created a new feature called is new. IF house built year > 2005 it is new and showed with 1.
#
# df["rooms+bathroom"] = df["bathrooms"]+ df["totalrooms"]
#
#
# X = df[["bathrooms","bedrooms","zindexvalue","finishedsqft","yearbuilt","price_per_sqft","is_new","high_price_low_freq","Duplex","MultiFamily2To4"]]
# y = df["lastsoldprice"]
#
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#
# d_train = lgb.Dataset(X_train, label=y_train)
#
# params = {
#     'boosting_type': 'gbdt',
#     'objective': 'regression',
#     'metric': 'rmsle',
#     'max_depth': 6,
#     'learning_rate': 0.1,
#     'verbose': 0}
# n_estimators = 100
#
# lgb_reg_model = lgb.train(params, d_train, num_boost_round = 100)
# y_tahmin = lgb_reg_model.predict(X_test)
# y_tahmin_train = lgb_reg_model.predict(X_train)
#
# rmse = np.sqrt(mean_squared_error(y_test, y_tahmin))
# print("RMSE: %f" % (rmse))
#
# plt.figure(figsize=(10,10))
# plt.title('Gerçek Değerler & Tahminler\n', size = 14)
# ax1 = plt.scatter(y_test, y_tahmin)
# ax2 = plt.scatter(y_train, y_tahmin_train,alpha=0.30)
# ax3 = plt.plot(y_test, y_test, color="red")
# plt.legend((ax1, ax2), ('Test Kümesi', 'Eğitim Kümesi'))
# plt.xlabel("Gerçek Değerler")
# plt.ylabel("Tahminler")
# plt.show()

######################################################################################################################
######################################################################################################################
######################################################################################################################

# Proje 3 Fraud credit card
# Classification
# import re
# from sklearn.metrics import accuracy_score
#
# df = pd.read_csv("creditcard_azaltılmış.csv")
#
# df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
#
# X = df.drop('Class', axis=1)
# y = df['Class']
#
# X_eğitim, X_test, y_eğitim, y_test = train_test_split(X, y, test_size=0.20, random_state=112)
#
# d_train = lgb.Dataset(X_eğitim, label=y_eğitim)
#
# params = {'boosting_type' : 'gbdt',
#           'objective' : 'binary',
#           'metric' : 'binary_logloss',
#           'sub_feature' : 0.5,
#           'num_leaves' :  10,
#           'min_data' : 50,
#           'max_depth' : 10}
#
# lgb_model = lgb.train(params, d_train, num_boost_round = 100)
#
# y_tahmin=lgb_model.predict(X_test)
# y_tahmin[:10]
#
#
# y_tahmin = [0 if tahmin < 0.5 else 1 for tahmin in y_tahmin]
# print(y_tahmin[:10])
#
# dogruluk=accuracy_score(y_tahmin,y_test)
# print(dogruluk)





















