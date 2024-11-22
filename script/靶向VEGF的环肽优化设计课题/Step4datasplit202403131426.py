import os
import time
import shutil
import random
import pickle

class config:

    def __init__(self):

        self.train = ["I3A","I3C","I3D","I3E","I3F","I3G","I3H","I3K","I3L","Z4A","Z4C","Z4D","Z4E","Z4F","Z4G","Z4H","Z4I","Z4K","Z4L","V5A","V5C","V5D","V5E","V5F","V5G","V5H","V5I","V5K","V5L","E8A","E8C","E8D","E8F","E8G","E8H","E8I","E8K","E8L","E13A","E13C","E13D","E13F","E13G","E13H","E13I","E13K","E13L",]
        self.predict = ["I3M","I3N","Z4M","Z4N","V5M","V5N","E8M","E8N","E13M","E13N",]

def train_lst(lst):

    all_train = []
    for i in lst:
        with open("input_vectors_" + i + ".pkl", "rb") as f:
            graphs_dict, labels = pickle.load(f)
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

    if not os.path.exists(os.path.join(".", "predict")):
        os.mkdir(os.path.join(".", "predict"))
    for i in settings.predict:
        shutil.copy("input_vectors_"+str(i)+".pkl", os.path.join(".", "predict", "input_vectors_"+str(i)+".pkl"))

    end = time.time()
    runtime_h = (end - start) / 3600
    print("run " + str(runtime_h) + " h.")

def main():
    run()
    
if __name__=="__main__":
    main() 