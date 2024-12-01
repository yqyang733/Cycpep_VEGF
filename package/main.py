import os
import time
import pickle
from concurrent.futures import ProcessPoolExecutor

from config import Config
from GetDescriptors import get_single_snapshot, read_complexes, remove_files, get_all_features, make_data
from TrainAndPredict import GB_params_adjust, GB_data_load, GB_model_train, GB_model_predict, GB_model_pretraindata

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
n_estimators         =       Config().n_estimators
max_depth            =       Config().max_depth
learning_rate        =       Config().learning_rate
subsample            =       Config().subsample
cpus                 =       Config().cpus

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

def get_every_descriptor(i):
    
    if os.path.exists(os.path.join("Descriptors", "input_vectors_" + i.split(",")[0] + ".pkl")):
        pass
    else:
        protein_list, ligand_list, des_path = get_single_snapshot(trajpath, i.split(",")[0], refstructure, traj, startframe, endframe, step, partA, partB)           
        graphs_dict = read_complexes(protein_list, ligand_list, des_path)
        all_features = get_all_features(graphs_dict)
        data = make_data(graphs_dict, all_features)

        with open(os.path.join("Descriptors", "input_vectors_" + i.split(",")[0] + ".pkl"), "wb") as f:
            pickle.dump((all_features, data), f) 

        remove_files(des_path)

def train_and_predict():

    train_lst = get_lst(trainlist)
    predict_lst = get_lst(predictlist)

    input_vec, train_names, train_feature, train_labels = GB_data_load(train_lst)
    params = GB_params_adjust(n_estimators, max_depth, learning_rate, subsample)

    for i in params:
        
        n_es = i[0]
        max_dep = i[1]
        lr = i[2]
        sample = i[3]

        model = GB_model_train(train_feature, train_labels, n_es, max_dep, lr, sample)
        GB_model_pretraindata(model, input_vec, n_es, max_dep, lr, sample)
        GB_model_predict(model, predict_lst, n_es, max_dep, lr, sample)

def run():
    
    start = time.time()

    mk_files()
        
    train_lst = get_lst(trainlist)
    with ProcessPoolExecutor(max_workers=int(cpus)) as executor:
        executor.map(get_every_descriptor, train_lst)

    # predict_lst = get_lst(predictlist)
    # with ProcessPoolExecutor(max_workers=int(cpus)) as executor:
    #     executor.map(get_every_descriptor, predict_lst)

    # train_and_predict()
        
    end = time.time()
    runtime_h = (end - start) / 3600
    print("run " + str(runtime_h) + " h.")

def main():
    run()
    
if __name__=="__main__":
    main() 