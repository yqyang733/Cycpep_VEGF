import os
import time
import pickle

from config import Config
from GetDescriptors import get_single_snapshot, read_complexes, remove_files, get_all_features
from GetFeatureMatrix import get_input
from TrainAndPredict import GB_model_train, GB_model_predict, GB_model_pretraindata

trajpath             =       Config().trajpath
refstructure         =       Config().refstructure
traj                 =       Config().traj
partA                =       Config().partA    
partB                =       Config().partB
startframe           =       Config().startframe
endframe             =       Config().endframe
step                 =       Config().step
trainlist            =       Config().trainlist
predictlist          =       Config().predictlist
pickdescriptorsways  =       Config().pickdescriptorsways
descriptornums       =       Config().descriptornums

def mk_files():

    if not os.path.exists("Descriptors"):
        os.makedirs("Descriptors")
    
    if not os.path.exists("results"):
        os.makedirs("results")

def get_lst(in_file):

    with open(in_file) as f:
        f1 = f.readlines()
    
    all_lst = []
    for i in f1:
        line = i.strip()
        all_lst.append(line)

    return all_lst

def get_every_descriptor(mode):
    
    if mode == "train":
        
        all_lst = get_lst(trainlist)
    
    elif mode == "predict":

        all_lst = get_lst(predictlist)

    for i in all_lst:
        
        if mode == "train":

            if os.path.exists(os.path.join("Descriptors", "Descriptors_" + i.split(",")[0] + ".pkl")):

                continue
            
            # elif os.path.exists(os.path.join("Descriptors", "predict_Descriptors_" + i.split(",")[0] + ".pkl")):

            #     with open(os.path.join("Descriptors", "predict_Descriptors_" + i.split(",")[0] + ".pkl"), "rb") as f:                    
            #         (graphs_dict, labels) = pickle.load(f)
            #     for i in labels.keys():
            #         labels[i] = float(i.split(",")[1])
            #     with open(os.path.join("Descriptors", "Descriptors_" + i.split(",")[0] + ".pkl"), "wb") as f:
            #         pickle.dump((graphs_dict, labels), f)
            #     f.close() 

            #     os.remove(os.path.join("Descriptors", "predict_Descriptors_" + i.split(",")[0] + ".pkl"))

            #     continue

            else:

                protein_list, ligand_list, labels_list = get_single_snapshot(trajpath, i.split(",")[0], float(i.split(",")[1]), refstructure, traj, startframe, endframe, step, partA, partB)
            
                graphs_dict, labels = read_complexes(protein_list, ligand_list, labels_list)
            
                with open(os.path.join("Descriptors", "Descriptors_" + i.split(",")[0] + ".pkl"), "wb") as f:
                    pickle.dump((graphs_dict, labels), f)

        elif mode == "predict":

            if os.path.exists(os.path.join("Descriptors", "Descriptors_" + i.split(",")[0] + ".pkl")):

                continue
            
        #     elif os.path.exists(os.path.join("Descriptors", "Descriptors_" + i.split(",")[0] + ".pkl")):

        #         os.rename(os.path.join("Descriptors", "Descriptors_" + i.split(",")[0] + ".pkl"), os.path.join("Descriptors", "predict_Descriptors_" + i.split(",")[0] + ".pkl"))

        #         continue

            else:

                protein_list, ligand_list, labels_list = get_single_snapshot(trajpath, i, 0, refstructure, traj, startframe, endframe, step, partA, partB)

                graphs_dict, labels = read_complexes(protein_list, ligand_list, labels_list)
                
                with open(os.path.join("Descriptors", "Descriptors_" + i.split(",")[0] + ".pkl"), "wb") as f:
                    pickle.dump((graphs_dict, labels), f) 

        remove_files(i.split(",")[0], refstructure, traj, startframe, endframe, step)

def get_all_descriptor():

    with open(trainlist) as f:
            f1 = f.readlines()
    
    train_lst = []
    for i in f1:
        line = i.strip().split(",")
        train_lst.append(line[0])

    get_all_features(train_lst)

def get_feature_matrix():

    all_lst = get_lst(trainlist)
    all_lst = [i.split(",")[0] for i in all_lst]
    all_lst += get_lst(predictlist)
    get_input(all_lst)

def train():

    all_lst = get_lst(trainlist)
    all_lst = [i.split(",")[0] for i in all_lst]

    model = GB_model_train(all_lst)

    GB_model_pretraindata(model, all_lst)

    return model

def predict(model):

    all_lst = get_lst(predictlist)

    GB_model_predict(all_lst, model)

def run():
    
    start = time.time()

    # mk_files()
    # get_every_descriptor("train")
    # get_every_descriptor("predict")
    # get_all_descriptor()
    # get_feature_matrix()
    model = train()
    predict(model)
        
    end = time.time()
    runtime_h = (end - start) / 3600
    print("run " + str(runtime_h) + " h.")

def main():
    run()
    
if __name__=="__main__":
    main() 