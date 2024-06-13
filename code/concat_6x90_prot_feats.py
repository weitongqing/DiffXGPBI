# 拼接phage和host的6 × 90的Protein特征
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re

def walkFile(file):
    phage_name = []
    for root, dirs, files in os.walk(file):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        # 遍历文件
        for f in files:
            phage_name.append(os.path.join(f))
    return phage_name


if __name__ == '__main__':
    walkfile = walkFile('/bios-store1/home/TongqingWei/phage/phage_external/PHIAF/protein_feature')
    # 创建所有phage为行索引，798特征为列的空表
    CDD_diff = list(range(540))
    ID_diff = []
    for ls in walkfile:
        if ls[-3:] == 'csv':
            ID_diff.append(ls[:-4])
    KP_df = pd.DataFrame(index = ID_diff,columns = CDD_diff)
    KP_df = KP_df.replace(np.nan,0)
    # 填充空表
    for ls in walkfile:
        if ls[-3:] == 'csv':
            KP = pd.read_csv("/bios-store1/home/TongqingWei/phage/phage_external/PHIAF/protein_feature/" + ls, header = 0)
            KP_diff = pd.DataFrame(index = ['mean', 'max', 'min', 'std', 'median', 'var'], columns = KP.columns[1:])
            KP_diff = KP_diff.replace(np.nan, 0)
            # 填充6 × 90的dataframe
            for i in KP.columns[1:]:
                KP_diff.loc['mean', i] = np.mean(KP[i])
                KP_diff.loc['max', i] = max(KP[i])
                KP_diff.loc['min', i] = min(KP[i])
                KP_diff.loc['std', i] = np.std(KP[i])
                KP_diff.loc['median', i] = np.median(KP[i])
                KP_diff.loc['var', i] = np.var(KP[i])
            KP_diff = pd.concat([KP_diff.iloc[0,:], KP_diff.iloc[1,:], KP_diff.iloc[2,:], KP_diff.iloc[3,:], KP_diff.iloc[4,:], KP_diff.iloc[5,:]])   
            # 将每个phage的798特征向量填入最终表
            for index in KP_df.index:
                if ls[:-4] == index:
                    KP_df.loc[index,:] = np.asarray(KP_diff)
    KP_df.to_csv("/bios-store1/home/TongqingWei/phage/phage_external/PHIAF/protein_feature/protein_features.csv")