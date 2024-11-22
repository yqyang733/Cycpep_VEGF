import os
import time
import pickle
import numpy as np
from scipy.stats import pearsonr
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

class config:

    def __init__(self):

        self.predict = ["I3M","I3N","Z4M","Z4N","V5M","V5N","E8M","E8N","E13M","E13N",]

def gb_model(predict):
    
    clf = GradientBoostingRegressor(n_estimators = 100, max_depth = 3, learning_rate = 0.05)

    train_feature = []
    train_labels = []
    
    with open(os.path.join(".", "train", "train.pkl"), "rb") as f:
        train = pickle.load(f)
    for a in train:
        train_feature.append(a[1])
        train_labels.append(a[2])
    
    train_feature = np.matrix(train_feature)
    rf = clf.fit(train_feature, train_labels)
    pred_train = rf.predict(train_feature)
    with open(os.path.join("train", "train_pre_labels.csv"), "w") as f:
        f.write("predict,labels\n")
        for a in range(len(train_labels)):
            f.write(str(pred_train[a])+","+str(train_labels[a])+"\n")
    mse_train = mean_squared_error(train_labels, pred_train)
    pearson_train = pearsonr(train_labels, pred_train)
    ci_train = concordance_index(train_labels, pred_train)
    with open(os.path.join("train", "train_metrics.txt"), "w") as f:
        f.write("mse,pearson,ci\n")
        f.write(str(mse_train)+","+str(pearson_train)+","+str(ci_train))
    
    for i in predict:
        predict_feature = []
        predict_labels = []
        rt_pre_label = open(os.path.join(".", "predict", "predict_pre_label_"+str(i)+".csv"), "w")
        rt_pre_label.write("predict,label\n")
        with open(os.path.join(".", "predict", "input_vectors_"+str(i)+".pkl"), "rb") as f:
            graphs_dict, labels = pickle.load(f)
        for j in graphs_dict.keys():
            predict_feature.append(graphs_dict[j])
        for j in labels:
            predict_labels.append(labels[j])
        predict_feature = np.matrix(predict_feature)
        pred_predict = rf.predict(predict_feature)
        for a in range(len(predict_labels)):
            rt_pre_label.write(str(pred_predict[a])+","+str(predict_labels[a])+"\n")
        
def run():

    start = time.time()

    settings = config()
    gb_model(settings.predict)

    end = time.time()
    runtime_h = (end - start) / 3600
    print("run " + str(runtime_h) + " h.")

def main():
    run()
    
if __name__=="__main__":
    main() 