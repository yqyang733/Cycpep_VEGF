import time
import pickle
import numpy as np

class config:

    def __init__(self):

        lst_tmp = list()
        with open("double_lst2") as f:
            f1 = f.readlines()

        for i in f1:
            lst_tmp.append(i.strip())

        self.features_lst = lst_tmp

def make_data(graphs_dict):    
    
    with open("Descriptors_sele.pkl", "rb") as f:
        selected_descriptors = pickle.load(f)
    
    data = dict()
    
    ''' For each complex '''
    for name in graphs_dict.keys():
        whole_descriptors = dict()
        
        for type in graphs_dict[name].keys():
            
            ''' 
            one descriptor check  
            e.g. (16, 16):[{(1, 16, 6, '1'): 2, (0, 16, 6, '1'): 1}, ...]    
            '''
            for descriptor in graphs_dict[name][type]:
                if tuple(sorted(descriptor.items())) in whole_descriptors:
                    whole_descriptors[tuple(sorted(descriptor.items()))] += 1
                else:
                    whole_descriptors[tuple(sorted(descriptor.items()))] = 1  
        
        ''' Create a row vector of size 2,500 for each complex. '''
        row_vetor = list()
        for selected_descriptor in selected_descriptors:
            row_vetor.append(whole_descriptors[selected_descriptor]) if selected_descriptor in whole_descriptors else row_vetor.append(0)
                
        data[name] = np.array(row_vetor, dtype = np.float32)    
        
    return data

def get_input(lst):

    for i in lst:
        #graphs_dict = dict()
        #labels = dict()
        with open("Descriptors_" + i + ".pkl", "rb") as f:
            graphs_dict, labels = pickle.load(f)
            #graphs_dict[i+"_1000_r/"+i+"_1000_l"] = graphs_dict_all[i+"_1000_r/"+i+"_1000_l"]
            #labels[i+"_1000_r/"+i+"_1000_l"] = labels_all[i+"_1000_r/"+i+"_1000_l"]
        data = make_data(graphs_dict)
        with open("input_vectors_" + i + ".pkl", "wb") as f:
            pickle.dump((data, labels), f) 

def run():

    start = time.time()

    settings = config()

    get_input(settings.features_lst)

    end = time.time()
    runtime_h = (end - start) / 3600
    print("run " + str(runtime_h) + " h.")

def main():
    run()
    
if __name__=="__main__":
    main() 
