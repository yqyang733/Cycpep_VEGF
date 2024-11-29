import os
import pickle
import random
import torch 
import shutil
import numpy as np

from config import Config
from model.GradientBoostingRegressor import GradientBoosting
from model.MLP import MLPNet
from model.CNN import CNNNet
from model.Transformer import TransformerNet

pickdescriptorsways  =       Config().pickdescriptorsways
descriptornums       =       Config().descriptornums
mlp_n_hidden         =       Config().n_hidden
epoch                =       Config().epoch
channels             =       Config().channels
noise                =       Config().noise

def mk_files(fle):

    if not os.path.exists(fle):
        os.makedirs(fle)
    else:
        shutil.rmtree(fle)
        os.makedirs(fle)

def split_into_groups(data, group_size):
    """
    将列表等分为组，每组间隔 group_size。
    
    参数：
        data (list): 原始列表
        group_size (int): 分组的步长
    
    返回：
        list: 分好的组，格式如 [[1,6], [2,7], ...]
    """
    return [[data[i] for i in range(j, len(data), group_size)] for j in range(group_size)]

def traindata_prepare(lst):

    all_vec_lst = []

    for i in lst:
        
        mut_name = i.replace("\n", "").split(",")[0]
        ddg = i.replace("\n", "").split(",")[1]

        with open(os.path.join("Descriptors", "input_vectors_" + mut_name + ".pkl"), "rb") as f:
            graphs_dict, labels = pickle.load(f)

        if noise:
            se = float(i.replace("\n", "").split(",")[2])
            print("se", se)
            labels_norm = np.random.normal(loc=float(ddg), scale=se, size=len(labels))
            for a in range(len(labels)):
                labels[list(labels.keys())[a]] = labels_norm[a]
        else:
            labels_dup = np.array([ddg] * len(labels), dtype=np.float64)
            for a in range(len(labels)):
                labels[list(labels.keys())[a]] = labels_dup[a]

        groups = split_into_groups(list(graphs_dict.keys()), int(len(list(graphs_dict.keys()))/channels))

        for b in groups:
            frame_names = ":".join(b)
            graph = []
            label = []
            for a in b:
                graph.append(graphs_dict[a])
                label.append(labels[a])
            label_mean = np.mean(label)
        
            all_vec_lst.append([frame_names, graph, label_mean])
    
    return all_vec_lst

def predictdata_prepare(lst):

    all_vec_lst = []

    for i in lst:
        with open(os.path.join("Descriptors", "input_vectors_" + i + ".pkl"), "rb") as f:
            graphs_dict, labels = pickle.load(f)

        groups = split_into_groups(list(graphs_dict.keys()), int(len(list(graphs_dict.keys()))/channels))

        for b in groups:
            frame_names = "/".join(b)
            graph = []
            label = []
            for a in b:
                graph.append(graphs_dict[a])
                label.append(labels[a])
            label_mean = np.mean(label)
        
            all_vec_lst.append([frame_names, graph, label_mean])
    
    return all_vec_lst

def select_descriptors_data(lst):

    input_vec = []

    if pickdescriptorsways == "frequency":
        if descriptornums == -1:
            input_vec = lst
        else:
            for i in lst:
                input_vec.append([i[0], np.array(i[1])[:range(descriptornums)], i[2]])

    elif pickdescriptorsways == "std":
        all_vec = np.array([row[1] for row in lst])
        all_vec = all_vec.reshape(-1, all_vec.shape[-1])
        stds = np.std(all_vec, axis=0)
        sorted_indices = np.argsort(stds)[::-1]

        if descriptornums == -1:
            for i in lst:
                input_vec.append(i[0], np.array(i[1])[:sorted_indices], i[2])
        else:
            for i in lst:
                input_vec.append(i[0], np.array(i[1])[:sorted_indices[:descriptornums]], i[2])

    elif pickdescriptorsways == "forward":
        feature_select_idx = forward_selection(lst)
        for i in lst:
            input_vec.append([i[0], i[1][feature_select_idx], i[2]])

    return input_vec

def GB_params_adjust(n_es, max_dep, lr, sample):

    all_params = []
    n_es = n_es.split(":")
    max_dep = max_dep.split(":")
    lr = lr.split(":")
    sample = sample.split(":")

    for a in n_es:
        for b in max_dep:
            for c in lr:
                for d in sample:
                    all_params.append([int(a),int(b),float(c),float(d)])

    return all_params

def GB_data_load(lst):

    train_names = []
    train_feature = []
    train_labels = []
    
    all_vec_lst = traindata_prepare(lst)
    # print("all_vec_lst", all_vec_lst)
    input_vec = select_descriptors_data(all_vec_lst)
    # print("input_vec", input_vec)
    random.shuffle(input_vec)

    for a in input_vec:
        train_names.append(a[0])
        train_feature.append(a[1])
        train_labels.append(a[2])
    
    # print("train_feature", train_feature)
    # print("train_labels", train_labels)
    
    train_feature = np.array([np.concatenate(sub_array).tolist() for sub_array in train_feature])  # 如果多个channel将多个channel的feature数组根据channel数目依次拼接。一个channel的所有feature接着下一个channel的所有feature。
    # train_feature = np.array([np.array(sub_array).T.flatten().tolist() for sub_array in train_feature])   # 如果多个channel将多个channel的fenture数组根据feature的次序进行拼接。第一个feature的所有channel接着下一个feature的所有channel。

    return input_vec, train_names, train_feature, train_labels

def GB_model_train(train_feature, train_labels, n_es, max_dep, lr, sample):

    clf = GradientBoosting(n_estimators = n_es, max_depth = max_dep, learning_rate = lr, subsample = sample).build()

    rf = clf.fit(train_feature, train_labels)

    return rf

def GB_model_predict(model, lst, n_es, max_dep, lr, sample):

    f_name = str(n_es) + "_" + str(max_dep) + "_" + str(lr) + "_" + str(sample)

    predict_rt = open(os.path.join("results", f_name, "all_individal_pre.csv"), "w")
    predict_rt.write("mut,mean_pred,se_pred\n")
    predict_confs_rt = open(os.path.join("results", f_name, "all_confs_pre.csv"), "w")
    predict_confs_rt.write("mut,value_pre\n")

    for i in lst:

        predict_feature = []
        predict_labels = []
        predict_names = []
        
        vec_every = predictdata_prepare([i,])
        input_vec = select_descriptors_data(vec_every)

        for a in input_vec:
            predict_names.append(a[0])
            predict_feature.append(a[1])
            predict_labels.append(a[2])

        predict_feature = np.array([np.concatenate(sub_array).tolist() for sub_array in predict_feature])  # 如果多个channel将多个channel的feature数组根据channel数目依次拼接。一个channel的所有feature接着下一个channel的所有feature。
        # predict_feature = np.array([np.array(sub_array).T.flatten().tolist() for sub_array in predict_feature])   # 如果多个channel将多个channel的fenture数组根据feature的次序进行拼接。第一个feature的所有channel接着下一个feature的所有channel。

        pred_predict = model.predict(predict_feature)

        pred_mean = np.mean(pred_predict)
        pred_se = np.std(pred_predict)
        predict_rt.write(i+","+str(pred_mean)+","+str(pred_se)+"\n")

        for i in range(len(predict_names)):
            predict_confs_rt.write(predict_names[i]+","+str(pred_predict[i])+"\n")
    
    predict_confs_rt.close()    
    predict_rt.close()

def GB_model_pretraindata(model, lst, n_es, max_dep, lr, sample):

    f_name = str(n_es) + "_" + str(max_dep) + "_" + str(lr) + "_" + str(sample)

    mk_files(os.path.join("results", f_name))

    predict_trainconfs_rt = open(os.path.join("results", f_name, "all_trainconfs_prelab.csv"), "w")
    predict_trainconfs_rt.write("mut,value_label,value_pre\n")

    predict_trainnames = []
    predict_trainfeature = []
    predict_trainlabels = []

    for a in lst:
        predict_trainnames.append(a[0])
        predict_trainfeature.append(a[1])
        predict_trainlabels.append(a[2])

    predict_trainfeature = np.array([np.concatenate(sub_array).tolist() for sub_array in predict_trainfeature])  # 如果多个channel将多个channel的feature数组根据channel数目依次拼接。一个channel的所有feature接着下一个channel的所有feature。
    # predict_trainfeature = np.array([np.array(sub_array).T.flatten().tolist() for sub_array in predict_trainfeature])   # 如果多个channel将多个channel的fenture数组根据feature的次序进行拼接。第一个feature的所有channel接着下一个feature的所有channel。

    pred_trainpredict = model.predict(predict_trainfeature)

    for i in range(len(predict_trainnames)):
        predict_trainconfs_rt.write(predict_trainnames[i]+","+str(predict_trainlabels[i])+","+str(pred_trainpredict[i])+"\n")
    predict_trainconfs_rt.close()

def forward_selection(all_vec_train):

    random.shuffle(all_vec_train)
    feature_nums = len(all_vec_train[0][1])
    all_vec = np.array([row[1] for row in all_vec_train])
    all_label = np.array(row[2] for row in all_vec_train)
    dict_feature_pcc = dict()
    feature_all_index = range(feature_nums)
    feature_select_index = list()
    pcc_final = -1000
    while len(feature_select_index) < len(feature_nums): 
        pcc_every_feature = -1000
        for i in feature_all_index:

            temp_feature_index = feature_select_index.append(i)
            trainall_feature = all_vec[:, temp_feature_index]

            assert len(trainall_feature) == len(all_label)

            pcc_all = []

            for _ in range(100):
                indices = list(range(len(trainall_feature)))
                random.shuffle(indices)

                # 选取 80% 的数据
                split_index = int(0.8 * len(trainall_feature))
                train_indices = indices[:split_index]
                validation_indices = indices[split_index:]

                # 根据索引创建新的训练和测试数组
                vec_train = trainall_feature[train_indices]
                label_train = all_label[train_indices]
                vec_validation = trainall_feature[validation_indices]
                label_validation = all_label[validation_indices]

                clf = GradientBoosting(n_estimators = n_estimators, max_depth = max_depth, learning_rate = learning_rate).build()

                train_feature = np.matrix(vec_train)
                rf = clf.fit(train_feature, label_train)
                validation_feature = np.matrix(vec_validation)
                pred_validation = rf.predict(validation_feature)

                pcc_matrix = np.corrcoef(pred_validation, label_validation)
                pcc_value = pcc_matrix[0, 1]
                pcc_all.append(pcc_value)

            pcc_mean = np.mean(np.array(pcc_all))
            pcc_std = np.std(np.array(pcc_all))

            if pcc_mean > pcc_every_feature:
                pcc_every_feature = pcc_mean
                pcc_every_feature_std = pcc_std
                every_feature_index = i

        dict_feature_pcc[feature_select_index.append(every_feature_index)] = (pcc_every_feature, pcc_every_feature_std)
        
        feature_select_index.append(every_feature_index)
        feature_all_index.remove(every_feature_index)
        
        if pcc_every_feature > pcc_final:
            pcc_final = pcc_every_feature
            pcc_final_std = pcc_every_feature_std
            feature_final = feature_select_index.append(every_feature_index)
    
    feature_pcc = open(os.path.join("results", "feature_pcc.csv"), "w")
    feature_pcc.write("Feature,PCC_mean,PCC_std\n")
    for i in dict_feature_pcc.keys():
        feature_pcc.write(i+","+dict_feature_pcc[i][0]+","+dict_feature_pcc[i][1]+"\n")
    feature_pcc.close()

    print("使得相关性最强的特征index是"+feature_final+"，其对应的PCC是"+str(pcc_final))
        
    return feature_final

def MLP_model_train(lst):

    train_feature = []
    train_labels = []
    
    all_vec_lst = traindata_prepare(lst)
    input_vec = select_descriptors_data(all_vec_lst)
    random.shuffle(input_vec)

    for a in input_vec:
        train_feature.append(a[1])
        train_labels.append(a[2])   # 数据应该需要归一化处理，这里还没有归一化，后面报错了注意加上归一化。

    train_feature = torch.tensor(train_feature)
    train_labels = torch.tensor(train_labels)

    Net = MLPNet(train_feature.shape[1], mlp_n_hidden, 1)

    optimizer = torch.optim.SGD(Net.parameters(), lr = 0.2)
    loss_func = torch.nn.MSELoss()

    for epo in range(epoch):
        prediction = Net(train_feature)
        loss = loss_func(prediction, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return Net

def MLP_model_predict(lst, model):

    predict_rt = open(os.path.join("results", "all_individal_pre.csv"), "w")

    for i in lst:

        predict_feature = []
        predict_labels = []
        
        vec_every = predictdata_prepare([i,])
        input_vec = select_descriptors_data(vec_every)

        for a in input_vec:
            predict_feature.append(a[1])
            predict_labels.append(a[2])

        predict_feature = torch.tensor(predict_feature)
        predict_labels = torch.tensor(predict_labels)

        prediction = model(predict_feature)

        predict_rt.write("mut,mean_pred,se_pred\n")
        pred_mean = np.mean(prediction.numpy())
        pred_se = np.std(prediction.numpy())
        predict_rt.write(i+","+str(pred_mean)+","+str(pred_se)+"\n")
        
    predict_rt.close()

def CNN_model_train(lst):

    train_feature = []
    train_labels = []
    
    all_vec_lst = traindata_prepare(lst)
    input_vec = select_descriptors_data(all_vec_lst)
    random.shuffle(input_vec)

    for a in input_vec:
        train_feature.append(a[1])
        train_labels.append(a[2])   # 数据应该需要归一化处理，这里还没有归一化，后面报错了注意加上归一化。

    train_feature = torch.tensor(train_feature)
    train_feature = train_feature.unsqueeze(1)
    train_labels = torch.tensor(train_labels)

    Net = CNNNet()

    optimizer = torch.optim.SGD(Net.parameters(), lr = 0.2)
    loss_func = torch.nn.MSELoss()

    for epo in range(epoch):
        prediction = Net(train_feature)
        loss = loss_func(prediction, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return Net

def CNN_model_predict(lst, model):

    predict_rt = open(os.path.join("results", "all_individal_pre.csv"), "w")

    for i in lst:

        predict_feature = []
        predict_labels = []
        
        vec_every = predictdata_prepare([i,])
        input_vec = select_descriptors_data(vec_every)

        for a in input_vec:
            predict_feature.append(a[1])
            predict_labels.append(a[2])

        predict_feature = torch.tensor(predict_feature)
        predict_feature = predict_feature.unsqueeze(1)
        predict_labels = torch.tensor(predict_labels)

        prediction = model(predict_feature)

        predict_rt.write("mut,mean_pred,se_pred\n")
        pred_mean = np.mean(prediction.numpy())
        pred_se = np.std(prediction.numpy())
        predict_rt.write(i+","+str(pred_mean)+","+str(pred_se)+"\n")
        
    predict_rt.close()

def Transformer_model_train(lst):

    train_feature = []
    train_labels = []
    
    all_vec_lst = traindata_prepare(lst)
    input_vec = select_descriptors_data(all_vec_lst)
    random.shuffle(input_vec)

    for a in input_vec:
        train_feature.append(a[1])
        train_labels.append(a[2])   # 数据应该需要归一化处理，这里还没有归一化，后面报错了注意加上归一化。

    train_feature = torch.tensor(train_feature)
    train_feature = train_feature.unsqueeze(2)
    train_feature = train_feature.permute(1, 0, 2)
    train_labels = torch.tensor(train_labels)

    Net = TransformerNet(input_dim=1, output_dim=1, seq_len=train_feature.shape[0])

    optimizer = torch.optim.SGD(Net.parameters(), lr = 0.2)
    loss_func = torch.nn.MSELoss()

    for epo in range(epoch):
        Net.train() 
        prediction = Net(train_feature)
        loss = loss_func(prediction, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return Net

def Transformer_model_predict(lst, model):

    predict_rt = open(os.path.join("results", "all_individal_pre.csv"), "w")

    for i in lst:

        predict_feature = []
        predict_labels = []
        
        vec_every = predictdata_prepare([i,])
        input_vec = select_descriptors_data(vec_every)

        for a in input_vec:
            predict_feature.append(a[1])
            predict_labels.append(a[2])

        predict_feature = torch.tensor(predict_feature)
        predict_feature = predict_feature.unsqueeze(2)
        predict_feature = predict_feature.permute(1, 0, 2)
        predict_labels = torch.tensor(predict_labels)

        model.eval()
        with torch.no_grad():
            prediction = model(predict_feature)

        predict_rt.write("mut,mean_pred,se_pred\n")
        pred_mean = np.mean(prediction.numpy())
        pred_se = np.std(prediction.numpy())
        predict_rt.write(i+","+str(pred_mean)+","+str(pred_se)+"\n")
        
    predict_rt.close()