import pandas as pd
import numpy as np
neg = pd.read_csv('/bios-store1/home/TongqingWei/PHI/Chinese/neg.txt', sep = '\t', header = 0)
neg

phage = pd.read_csv('/bios-store1/home/TongqingWei/PHI/Chinese/phage_external_feature.csv', index_col = 0)
phage

host = pd.read_csv('/bios-store1/home/TongqingWei/PHI/Chinese/host_external_feature.csv', index_col = 0)
host

fea = []
for i in range(len(neg)):
    fea.append(phage.loc[neg.iloc[i,0]].values - host.loc[neg.iloc[i,1][:-2]].values)
fea = np.asarray(fea)
fea

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components = 1570)
gmm.fit(fea)
predict = gmm.predict(fea)

pd.DataFrame(predict).to_csv('/bios-store1/home/TongqingWei/PHI/Chinese/label.csv', index = False)
pd.DataFrame(gmm.means_).to_csv('/bios-store1/home/TongqingWei/PHI/Chinese/center.csv', index = False)