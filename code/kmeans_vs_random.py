import pandas as pd
km = pd.read_excel('E:/公共数据/中文期刊/rawdata/train_set_kmeans.xlsx', sheet_name = 'data', header = 0)
rd = pd.read_excel('E:/公共数据/中文期刊/rawdata/train_set_random.xlsx', sheet_name = 'data', header = 0)

km = km.iloc[:,:-1]
rd = rd.iloc[:,:-1]
km

import seaborn as sns
# 不同噬菌体tSNE降维可视化
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['pdf.fonttype'] = 42 
rcParams['ps.fonttype'] = 42

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_style(style = 'white')
#palette = sns.color_palette("Set2", 1)
#plt.figure(figsize=(6,8))
from sklearn.manifold import TSNE
import seaborn as sns
#point_total = point + list(set(point_plus))
stdScale = StandardScaler().fit(km)
feature = stdScale.transform(km)
X = feature
#X = np.asarray(fea)
labelArray = ['positive'] * 1570 + ['negtive'] * 1570
y = labelArray
tsne = TSNE()
X_embedded = tsne.fit_transform(X)

palette = sns.color_palette("bright", 2)
plt.xlabel('t-SNE1', fontsize = 15)
plt.ylabel('t-SNE2', fontsize = 15)
plt.title('kmeans', fontsize = 18)

ax = sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, palette=palette)
plt.setp(ax.get_legend().get_texts(), fontsize = '15')
plt.show()
#sns.set_style('white')
#plt.savefig('C:/Users/admin/Desktop/kmeans.pdf', dpi = 400)
#sns.scatterplot(X_embedded[:4998,0], X_embedded[:4998,1])
#plt.scatter(X_embedded[:4998,0], X_embedded[:4998,1])

from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['pdf.fonttype'] = 42 
rcParams['ps.fonttype'] = 42

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_style(style = 'white')
#palette = sns.color_palette(['#1f78b4', '#ff7f00'])
palette = sns.color_palette("bright", 2)
plt.xlabel('t-SNE1', fontsize = 15)
plt.ylabel('t-SNE2', fontsize = 15)
plt.title('kmeans', fontsize = 18)


ax = sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, palette=palette)
plt.setp(ax.get_legend().get_texts(), fontsize = '15')
plt.savefig('E:/公共数据/中文期刊/result/kmeans.pdf', dpi = 400)
plt.show()

import seaborn as sns
# 不同噬菌体tSNE降维可视化
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['pdf.fonttype'] = 42 
rcParams['ps.fonttype'] = 42

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set_style('white')
#palette = sns.color_palette("Set2", 1)
#plt.figure(figsize=(6,8))
from sklearn.manifold import TSNE
import seaborn as sns
#point_total = point + list(set(point_plus))
stdScale = StandardScaler().fit(rd)
feature = stdScale.transform(rd)
X = feature
#X = np.asarray(fea)
labelArray = ['positive'] * 1570 + ['negtive'] * 1570
y = labelArray
tsne = TSNE()
X_embedded_rd = tsne.fit_transform(X)
X_embedded_rd = np.concatenate((X_embedded[:1211,:], X_embedded_rd[1211:,:]), axis = 0)

palette = sns.color_palette("bright", 2)
plt.xlabel('t-SNE1', fontsize = 15)
plt.ylabel('t-SNE2', fontsize = 15)
plt.title('random', fontsize = 18)
ax = sns.scatterplot(X_embedded_rd[:,0], X_embedded_rd[:,1], hue=y, palette=palette)
plt.setp(ax.get_legend().get_texts(), fontsize = '15')
plt.show()
#sns.set_style('white')
#plt.savefig('C:/Users/admin/Desktop/random.pdf', dpi = 400)
#sns.set_style('white')
#plt.savefig('C:/Users/admin/Desktop/TCR/random.jpg', dpi = 400)
#sns.scatterplot(X_embedded[:4998,0], X_embedded[:4998,1])
#plt.scatter(X_embedded[:4998,0], X_embedded[:4998,1])

from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['pdf.fonttype'] = 42 
rcParams['ps.fonttype'] = 42

#palette = sns.color_palette(['#1f78b4', '#ff7f00'])
palette = sns.color_palette("bright", 2)
plt.xlabel('t-SNE1', fontsize = 15)
plt.ylabel('t-SNE2', fontsize = 15)
plt.title('random', fontsize = 18)
ax = sns.scatterplot(X_embedded_rd[:,0], X_embedded_rd[:,1], hue=y, palette=palette)
plt.setp(ax.get_legend().get_texts(), fontsize = '15')
plt.savefig('E:/公共数据/中文期刊/result/random.pdf', dpi = 400)
plt.show()

print(np.var(X_embedded[1570:,0]), np.var(X_embedded[1570:,1]))

print(np.var(X_embedded_rd[1570:,0]), np.var(X_embedded_rd[1570:,1]))

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['pdf.fonttype'] = 42 
rcParams['ps.fonttype'] = 42

fig, axes = plt.subplots(1,2)

#plt.subplots(121)
plt.figure(figsize=(10,6))
sns.set(rc={'figure.figsize':(12.7,6.27)})
sns.set_style('white')
#ns.set_style("dark", {"axes.facecolor": "#e9f3ea"})
g=sns.distplot(X_embedded[1211:,0],
               hist=False,
               kde=True,#开启核密度曲线kernel density estimate (KDE)
               kde_kws={'linestyle':'--','linewidth':'1','color':'#098154',#设置外框线属性
                        'shade':True,#开启填充                        
                       },label = 'kmeans(955.183)', ax = axes[0]
              )
#plt.savefig('C:/Users/admin/Desktop/TCR/km_0.jpg', dpi = 400)
f=sns.distplot(X_embedded_rd[1211:,0],
               hist=False,
               kde=True,#开启核密度曲线kernel density estimate (KDE)
               kde_kws={'linestyle':'--','linewidth':'1','color':'r', #设置外框线属性
                        'shade':True,#开启填充                        
                       },label = 'random(864.0191)', ax = axes[0]
              )
g.legend(prop = {'size': 12})
g.set_xlabel('t-SNE1', fontsize = '18')
g.set_title('kmeans vs random kernal density estimation', fontsize = '20')
#ns.set(rc={'figure.figsize':(8.7,8.27)})
#ns.set_style('white')
#ns.set_style("dark", {"axes.facecolor": "#e9f3ea"})
p=sns.distplot(X_embedded[1211:,1],
               hist=False,
               kde=True,#开启核密度曲线kernel density estimate (KDE)
               kde_kws={'linestyle':'--','linewidth':'1','color':'#098154',#设置外框线属性
                        'shade':True,#开启填充                        
                       },label = 'kmeans(1390.6318)', ax = axes[1]
              )
#plt.savefig('C:/Users/admin/Desktop/TCR/km_0.jpg', dpi = 400)
q=sns.distplot(X_embedded_rd[1211:,1],
               hist=False,
               kde=True,#开启核密度曲线kernel density estimate (KDE)
               kde_kws={'linestyle':'--','linewidth':'1','color':'r', #设置外框线属性
                        'shade':True,#开启填充                        
                       },label = 'random(1301.1256)', ax = axes[1]
              )
q.legend(prop = {'size': 12})
#.set_xticklabels(fontsize = 15)
q.set_xlabel('t-SNE2', fontsize = '18')
q.set_title('kmeans vs random kernal density estimation', fontsize = '20')
fig.set_figwidth(20)
fig.savefig('E:/公共数据/中文期刊/result/kmeans_random_KDE.pdf', dpi = 400)

import numpy as np
#X_embedded = np.asarray(km)
#X_embedded_rd = np.asarray(rd)
point = []
for i in range(1570,3140):
    ls = {}
    for j in range(3140):
        ls[j] = np.linalg.norm(X_embedded[i] - X_embedded[j], ord = 2)
    ls_order = sorted(ls.items(), key = lambda x:x[1], reverse = False)
    need = ls_order[1:6]
    n = 0
    for k in need:
        if k[0] < 1570:
            n += 1
    point.append(n)

point_rd = []
for i in range(1570,3140):
    ls = {}
    for j in range(3140):
        ls[j] = np.linalg.norm(X_embedded_rd[i] - X_embedded_rd[j], ord = 2)
    ls_order = sorted(ls.items(), key = lambda x:x[1], reverse = False)
    need = ls_order[1:6]
    n = 0
    for k in need:
        if k[0] < 1570:
            n += 1
    point_rd.append(n)

from matplotlib import rcParams
from matplotlib import pyplot as plt
import seaborn as sns
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['pdf.fonttype'] = 42 
rcParams['ps.fonttype'] = 42

fig, axes = plt.subplots(1,2)
sns.set_style('white')
#plt.subplots(121)
plt.figure(figsize=(15,6))
sns.set(rc={'figure.figsize':(12.7,6.27)})
sns.set_style('white')
#ns.set_style("dark", {"axes.facecolor": "#e9f3ea"})
g=sns.countplot(point, ax = axes[0], alpha = 0.8, palette = 'hls'
              )
#plt.savefig('C:/Users/admin/Desktop/TCR/km_0.jpg', dpi = 400)
#g.legend(prop = {'size': 12})
g.set_ylabel('count', fontsize = '18')
g.set_title('kmeans negtive sample density distribution', fontsize = '20')
#ns.set(rc={'figure.figsize':(8.7,8.27)})
#ns.set_style('white')
#ns.set_style("dark", {"axes.facecolor": "#e9f3ea"})
#plt.savefig('C:/Users/admin/Desktop/TCR/km_0.jpg', dpi = 400)
q=sns.countplot(point_rd, ax = axes[1], alpha = 0.8, palette = 'hls'
              )
#q.legend(prop = {'size': 12})
#.set_xticklabels(fontsize = 15)
q.set_ylabel('count', fontsize = '18')
q.set_title('random negtive sample density distribution', fontsize = '20')
fig.set_figwidth(20)
#fig.savefig('E:/公共数据/neg_sample_density_distribution.pdf', dpi = 400)

pd.DataFrame(X_embedded).to_csv('E:/公共数据/中文期刊/rawdata/km_tsne_axis.csv', index = False)
pd.DataFrame(X_embedded_rd).to_csv('E:/公共数据/中文期刊/rawdata/rd_tsne_axis.csv', index = False)

len(point + point_rd)

ex = pd.DataFrame(point + point_rd)
ex['label'] = ['K-means'] * 1570 + ['Random'] * 1570
ex.columns = ['sample', 'label']
ex

from matplotlib import rcParams
from matplotlib import pyplot as plt
import seaborn as sns
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['pdf.fonttype'] = 42 
rcParams['ps.fonttype'] = 42

fig, axes = plt.subplots(1,1)
sns.set_style('white')
#plt.subplots(121)
plt.figure(figsize=(55,4))
sns.set(rc={'figure.figsize':(12.7,6.27)})
sns.set_style('white')
#ns.set_style("dark", {"axes.facecolor": "#e9f3ea"})
g=sns.countplot(x = 'sample', hue = 'label', data = ex, ax = axes, alpha = 0.8, palette = 'hls'
              )
#plt.savefig('C:/Users/admin/Desktop/TCR/km_0.jpg', dpi = 400)
g.legend(prop = {'size': 12})
g.set_ylabel('count', fontsize = '18')
g.set_title('negtive sample density distribution', fontsize = '20')
# #ns.set(rc={'figure.figsize':(8.7,8.27)})
# #ns.set_style('white')
# #ns.set_style("dark", {"axes.facecolor": "#e9f3ea"})
# #plt.savefig('C:/Users/admin/Desktop/TCR/km_0.jpg', dpi = 400)
# q=sns.countplot(point_rd, ax = axes[1], alpha = 0.8, palette = 'hls'
#               )
# #q.legend(prop = {'size': 12})
# #.set_xticklabels(fontsize = 15)
# q.set_ylabel('count', fontsize = '18')
# q.set_title('random negtive sample density distribution', fontsize = '20')
fig.set_figwidth(10)
fig.savefig('E:/公共数据/中文期刊/result/neg_sample_density_distribution_combine.pdf', dpi = 400)