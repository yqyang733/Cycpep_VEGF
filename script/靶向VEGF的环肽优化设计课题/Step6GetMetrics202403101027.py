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

        dict_tmp = dict()
        with open("double_lst") as f:
            f1 = f.readlines()
        for i in f1:
            dict_tmp[i.strip()] = "0_0"

        self.predict = dict_tmp

def obtain_metrics(predict):

    metrics_every_mut = open(os.path.join(".", "predict", "metrics_every_mut.txt"), "w")
    metrics_average = open(os.path.join(".", "predict", "metrics_average.txt"), "w")
    predict_pred_lab = open(os.path.join(".", "predict", "individal_pred_lab.csv"), "w")
    predict_pred_lab.write("mut,mean_pred,mean_lab,se_pred,se_lab\n")

    all_mean_pred = []
    all_se_pred = []
    all_mean_true = []
    all_se_true = []

    for i in predict.keys():

        true_mean = float(predict[i].split("_")[0])
        true_se = float(predict[i].split("_")[1])
        
        pred_lst = []
        label = []
        
        with open(os.path.join(".", "predict", "predict_pre_label_"+str(i)+".csv")) as f:
            f.readline()
            pred_labels = f.readlines()

        for j in pred_labels:
            pred_lst.append(float(j.split(",")[0]))
            label.append(float(j.split(",")[1].replace("\n", "")))

        mse_pred = mean_squared_error(label, pred_lst)
        pearson_pred = pearsonr(label, pred_lst)
        ci_pred = concordance_index(label, pred_lst)

        metrics_every_mut.write(i+"\n")
        metrics_every_mut.write("MSE: "+str(mse_pred)+"\n")
        metrics_every_mut.write("Pearson: "+str(pearson_pred)+"\n")
        metrics_every_mut.write("CI: "+str(ci_pred)+"\n\n")

        pred_mean = np.mean(pred_lst)
        pred_se = np.std(pred_lst)/499

        all_mean_pred.append(pred_mean)
        all_se_pred.append(pred_se)
        all_mean_true.append(true_mean)
        all_se_true.append(true_se)

        predict_pred_lab.write(i+","+str(pred_mean)+","+str(true_mean)+","+str(pred_se)+","+str(true_se)+"\n")

    mse_mean = mean_squared_error(all_mean_true, all_mean_pred)
    pearson_mean = pearsonr(all_mean_true, all_mean_pred)
    ci_mean = concordance_index(all_mean_true, all_mean_pred)

    metrics_average.write("Mean: "+"\n")
    metrics_average.write("MSE: "+str(mse_mean)+"\n")
    metrics_average.write("Pearson: "+str(pearson_mean)+"\n")
    metrics_average.write("CI: "+str(ci_mean)+"\n\n")

    mse_se = mean_squared_error(all_se_true, all_se_pred)
    pearson_se = pearsonr(all_se_true, all_se_pred)
    ci_se = concordance_index(all_se_true, all_se_pred)

    metrics_average.write("Se: "+"\n")
    metrics_average.write("MSE: "+str(mse_se)+"\n")
    metrics_average.write("Pearson: "+str(pearson_se)+"\n")
    metrics_average.write("CI: "+str(ci_se)+"\n\n")

def run():

    start = time.time()

    settings = config()
    obtain_metrics(settings.predict)

    end = time.time()
    runtime_h = (end - start) / 3600
    print("run " + str(runtime_h) + " h.")

def main():
    run()
    
if __name__=="__main__":
    main() 