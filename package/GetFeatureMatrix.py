import os
import pickle
import numpy as np

def make_data(graphs_dict):    
    
    with open(os.path.join("Descriptors", "Descriptors_all.pkl"), "rb") as f:
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
        
        ''' Create a row vector for each complex. '''
        row_vetor = list()
        for selected_descriptor in selected_descriptors:
            row_vetor.append(whole_descriptors[selected_descriptor]) if selected_descriptor in whole_descriptors else row_vetor.append(0)
                
        data[name] = np.array(row_vetor, dtype = np.float32)    
        
    return data

def get_input(lst):

    for i in lst:
        with open(os.path.join("Descriptors", "Descriptors_" + i + ".pkl"), "rb") as f:
            graphs_dict, labels = pickle.load(f)
        data = make_data(graphs_dict)
        with open(os.path.join("Descriptors", "input_vectors_" + i + ".pkl"), "wb") as f:
            pickle.dump((data, labels), f) 
