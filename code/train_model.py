train = pd.read_excel('E:/公共数据/中文期刊/rawdata/validation_set.xlsx', sheet_name = 'data', header = 0)
train

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
train_data, test_data, train_label, test_label = train_test_split(train.iloc[:,:-1], train.iloc[:,-1], test_size=.3, random_state=123)
#stdScale = StandardScaler().fit(train_data)
#train_data = stdScale.transform(train_data)
#test_data = stdScale.transform(test_data)

import xgboost as xgb
xlf = xgb.XGBClassifier(max_depth = 50, random_state = 42, use_label_encoder=False, eval_metric = 'logloss', learning_rate = 0.01, n_estimators = 120, subsample = 0.9, objective = 'binary:logistic')
xlf.fit(train_data, train_label)
xlf.predict(test_data)
tra_label=xlf.predict(train_data) #训练集的预测标签
tes_label=xlf.predict(test_data)#测试集的预测标签
#label = data['label']
tes_prob = xlf.predict_proba(test_data)[:,1]
print("训练集：", accuracy_score(train_label,tra_label))
print("测试集：", accuracy_score(test_label,tes_label))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve
from sklearn.metrics import roc_curve, auc
print("训练集：", accuracy_score(train_label,tra_label))
print("测试集：", accuracy_score(test_label,tes_label))

tes_prob

fpr, tpr, thresholds_keras = roc_curve(test_label,tes_prob)
roc_auc = auc(fpr, tpr)
print("AUC : ", roc_auc)

sklearn_accuracy = accuracy_score(test_label, tes_label)
sklearn_precision = precision_score(test_label, tes_label, average='macro')
sklearn_recall = recall_score(test_label, tes_label, average='macro')
sklearn_f1 = f1_score(test_label, tes_label, average='macro')
print("[sklearn_metrics] accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1))

from sklearn.metrics import precision_recall_curve
precision, recall, _thresholds = precision_recall_curve(test_label,tes_prob)
roc_auc = auc(recall, precision)
print("AUPR : ", roc_auc)

tes_label=xlf.predict(train.iloc[:,:-1])#测试集的预测标签
#label = data['label']
test_label = train.iloc[:,-1]
tes_prob = xlf.predict_proba(train.iloc[:,:-1])[:,1]
#print("训练集：", accuracy_score(train_label,tra_label))
print("测试集：", accuracy_score(test_label,tes_label))

import pickle
pickle.dump(xlf, open("E:/公共数据/中文期刊/result/model.pickle.dat", 'wb'))
load_model = pickle.load(open("E:/公共数据/中文期刊/result/model.pickle.dat", "rb"))