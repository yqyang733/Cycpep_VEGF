import os
import time
import shutil
import random
import pickle

class config:

    def __init__(self):

        self.train = dict()
        with open("single.csv") as f:
            f1 = f.readlines()
        for i in f1:
            line = i.strip().split(",")
            self.train[line[0]] = float(line[1])

        with open("double.csv") as f:
            f1 = f.readlines()
        for i in f1:
            line = i.strip().split(",")
            self.train[line[0]] = float(line[1])
        
        print(self.train)

def is_file_in_folder(folder_path):

    # 使用 os.path.join() 来构建文件的完整路径
    file_path = os.path.join(folder_path)
    
    # 检查文件是否存在
    return os.path.exists(file_path)

def train_lst(lst):

    all_train = []
    for i in lst.keys():
        if is_file_in_folder("../../double_ddg_predict/run1/ML_single_predict_double/train_and_predict/predict/input_vectors_" + i + ".pkl"):
            with open(os.path.join("../../double_ddg_predict/run1/ML_single_predict_double/train_and_predict/predict/input_vectors_" + i + ".pkl"), "rb") as f:
                graphs_dict, labels = pickle.load(f)
        elif is_file_in_folder("../../double_ddg_predict/run1/ML_single_predict_double/train_and_predict_2/predict/input_vectors_" + i + ".pkl"):
            with open(os.path.join("../../double_ddg_predict/run1/ML_single_predict_double/train_and_predict_2/predict/input_vectors_" + i + ".pkl"), "rb") as f:
                graphs_dict, labels = pickle.load(f)
        else:
            with open(os.path.join("../../single_ddg_train_predict/MD_ML/Score/input_vectors_" + i + ".pkl"), "rb") as f:
                graphs_dict, labels = pickle.load(f)
        
        labels_tmp = dict()
        for a in labels.keys():
            labels_tmp[a] = lst[i]
        labels = labels_tmp
        print(labels)
        for j in graphs_dict.keys():
            all_train.append([j, graphs_dict[j], labels[j]])
    
    return all_train

def run():

    start = time.time()

    settings = config()

    all_train = train_lst(settings.train)
    random.shuffle(all_train)
    
    if not os.path.exists(os.path.join(".", "train")):
        os.mkdir(os.path.join(".", "train"))
    train_pkl = open(os.path.join(".", "train", "train.pkl"), "wb")
    pickle.dump(all_train, train_pkl) 

    # if not os.path.exists(os.path.join(".", "predict")):
    #     os.mkdir(os.path.join(".", "predict"))
    # for i in settings.predict:
    #     shutil.copy("input_vectors_"+str(i)+".pkl", os.path.join(".", "predict", "input_vectors_"+str(i)+".pkl"))

    end = time.time()
    runtime_h = (end - start) / 3600
    print("run " + str(runtime_h) + " h.")

def main():
    run()
    
if __name__=="__main__":
    main() 
