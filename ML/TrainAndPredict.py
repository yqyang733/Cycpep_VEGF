import os
import pickle
import random
import torch 
import shutil
import numpy as np
from collections import defaultdict

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

def reorganize_array(graphs, titles, all_titles):

    # 创建标题到索引的映射
    title_to_index = {title: idx for idx, title in enumerate(all_titles)}
    graphs = np.array(graphs)

    # 创建一个新的数组，初始为0
    new_shape = (graphs.shape[0], graphs.shape[1], len(all_titles)) 
    new_array = np.zeros(new_shape)
    
    # 遍历原数组标题，填充到新数组中
    for col_idx, title in enumerate(titles):
        if title in title_to_index:
            new_idx = title_to_index[title] 
            new_array[:, :, new_idx] = graphs[:, :, col_idx] 
    
    return new_array

def data_prepare(trainlst, predictlst):

    trainlst = [i.replace("\n", "")+",train" for i in trainlst]
    predictlst = [i.replace("\n", "")+",0.01,0.01,predict" for i in predictlst]

    train_all_vec_everymut = []
    predict_all_vec_everymut = []
    lst = trainlst + predictlst
    # all_titles = set()
    all_titles = np.array([])

    for i in lst:
        
        mut_name = i.split(",")[0]
        ddg = i.split(",")[1]
        labels = dict()
        flag = i.split(",")[-1]

        with open(os.path.join("Descriptors", "input_vectors_" + mut_name + ".pkl"), "rb") as f:
            all_features, graphs_dict = pickle.load(f)
        
        # all_features = np.array(list(all_features), dtype=object)
        print("all_features:", all_features)
        print("len(all_features)", len(all_features))

        if flag == "train":
            # all_titles.update(all_features)
            all_titles = np.concatenate((all_titles, all_features), axis=0)

        if noise:
            se = float(i.split(",")[2])
            labels_norm = np.random.normal(loc=float(ddg), scale=se, size=len(graphs_dict))
            for a in range(len(graphs_dict)):
                labels[list(graphs_dict.keys())[a]] = labels_norm[a]
        else:
            labels_dup = np.array([ddg] * len(graphs_dict), dtype=np.float32)
            for a in range(len(graphs_dict)):
                labels[list(graphs_dict.keys())[a]] = labels_dup[a]

        groups = split_into_groups(list(graphs_dict.keys()), int(len(list(graphs_dict.keys()))/channels))

        names = []
        graphs = []
        label_indivial = []
        for b in groups:
            if flag == "train":
                names.append(":".join(b))
            elif flag == "predict":
                names.append(mut_name)
            graph = []
            label = []
            for a in b:
                graph.append(graphs_dict[a])
                label.append(labels[a])
            label_mean = np.mean(label)

            graphs.append(graph)
            label_indivial.append(label_mean)

        if flag == "train":
            train_all_vec_everymut.append([names, all_features, graphs, label_indivial])
        elif flag == "predict":
            predict_all_vec_everymut.append([names, all_features, graphs, label_indivial])

    all_titles = set(all_titles)
    # all_titles = np.array(list(all_titles), dtype=object)
    print("all_titles", all_titles)
    print("len(all_titles)", len(all_titles))
    train_all_vec_lst = []
    predict_all_vec_lst = []

    for names, all_features, graphs, label_indivial in train_all_vec_everymut:

        graphs = reorganize_array(graphs, all_features, all_titles)
        print("graphs[:5]", graphs[:5])
        for bb in range(len(names)):
            train_all_vec_lst.append([names[bb], graphs[bb], label_indivial[bb]])

    for names, all_features, graphs, label_indivial in predict_all_vec_everymut:

        graphs = reorganize_array(graphs, all_features, all_titles)
        for bb in range(len(names)):
            predict_all_vec_lst.append([names[bb], graphs[bb], label_indivial[bb]])
    
    return train_all_vec_lst, predict_all_vec_lst

def select_descriptors_data(lst, indices, mode):

    input_vec = []

    if pickdescriptorsways == "frequency":
        if mode == "train":
            all_vec = np.array([row[1] for row in lst])
            all_vec = all_vec.reshape(-1, all_vec.shape[-1])
            print("all_vec", all_vec)
            print("all_vec[:5]", all_vec[:5])
            print("all_vec[1000:1005]", all_vec[1000:1005])
            print("all_vec[2000:2005]", all_vec[2000:2005])
            print("all_vec[3000:3005]", all_vec[3000:3005])
            print("all_vec[4000:4005]", all_vec[4000:4005])
            print("all_vec[5000:5005]", all_vec[5000:5005])
            print("all_vec[6000:6005]", all_vec[6000:6005])
            print("all_vec[7000:7005]", all_vec[7000:7005])
            print("all_vec[8000:8005]", all_vec[8000:8005])
            print("all_vec[9000:9005]", all_vec[9000:9005])
            fre = np.sum(all_vec, axis=0)
            print("fre ", fre)
            sorted_indices = np.argsort(fre)[::-1]
            print("sorted_indices", sorted_indices)
        elif mode == "predict":
            sorted_indices = indices

        if descriptornums == -1:
            for i in lst:
                # print("i[1]", i[1][:3])
                print("np.array(i[1])[:,sorted_indices]", np.array(i[1])[:,sorted_indices][:3])
                input_vec.append([i[0], np.array(i[1])[:,sorted_indices], i[2]])
        else:
            for i in lst:
                input_vec.append([i[0], np.array(i[1])[:,sorted_indices[:descriptornums]], i[2]])

    elif pickdescriptorsways == "std":
        all_vec = np.array([row[1] for row in lst])
        all_vec = all_vec.reshape(-1, all_vec.shape[-1])
        stds = np.std(all_vec, axis=0)
        sorted_indices = np.argsort(stds)[::-1]

        if descriptornums == -1:
            for i in lst:
                input_vec.append(i[0], np.array(i[1])[:,:,sorted_indices], i[2])
        else:
            for i in lst:
                input_vec.append(i[0], np.array(i[1])[:,:,sorted_indices[:descriptornums]], i[2])

    elif pickdescriptorsways == "forward":
        feature_select_idx = forward_selection(lst)
        for i in lst:
            input_vec.append([i[0], i[1][feature_select_idx], i[2]])

    return input_vec, sorted_indices

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

def GB_data_load(trainlst, predictlst):

    train_names = []
    train_feature = []
    train_labels = []
    
    train_all_vec_lst, predict_all_vec_lst = data_prepare(trainlst, predictlst)
    train_input_vec, sorted_indices = select_descriptors_data(train_all_vec_lst, "_", "train")
    print("train_input_vec[:5]", train_input_vec[:5])
    print("train_input_vec[1000:1005]", train_input_vec[1000:1005])
    print("train_input_vec[2000:2005]", train_input_vec[2000:2005])
    print("train_input_vec[3000:3005]", train_input_vec[3000:3005])
    print("train_input_vec[4000:4005]", train_input_vec[4000:4005])
    print("train_input_vec[5000:5005]", train_input_vec[5000:5005])
    print("train_input_vec[6000:6005]", train_input_vec[6000:6005])
    print("train_input_vec[7000:7005]", train_input_vec[7000:7005])
    print("train_input_vec[8000:8005]", train_input_vec[8000:8005])
    print("train_input_vec[9000:9005]", train_input_vec[9000:9005])

    random.shuffle(train_input_vec)

    predict_input_vec, _ = select_descriptors_data(predict_all_vec_lst, sorted_indices, "predict")
    print("predict_input_vec", predict_input_vec[:5])

    for a in train_input_vec:
        train_names.append(a[0])
        train_feature.append(a[1])
        train_labels.append(a[2])
    
    train_feature = np.array([np.concatenate(sub_array).tolist() for sub_array in train_feature], dtype=np.float32)  # 如果多个channel将多个channel的feature数组根据channel数目依次拼接。一个channel的所有feature接着下一个channel的所有feature。
    # train_feature = np.array([np.array(sub_array).T.flatten().tolist() for sub_array in train_feature])   # 如果多个channel将多个channel的fenture数组根据feature的次序进行拼接。第一个feature的所有channel接着下一个feature的所有channel。

    return train_input_vec, train_names, train_feature, train_labels, predict_input_vec

def GB_model_train(train_feature, train_labels, n_es, max_dep, lr, sample):

    clf = GradientBoosting(n_estimators = n_es, max_depth = max_dep, learning_rate = lr, subsample = sample).build()

    rf = clf.fit(train_feature, train_labels)

    return rf

def GB_model_predict(model, predict_input_vec, n_es, max_dep, lr, sample):

    f_name = str(n_es) + "_" + str(max_dep) + "_" + str(lr) + "_" + str(sample)

    predict_rt = open(os.path.join("results", f_name, "all_individal_pre.csv"), "w")
    predict_rt.write("mut,mean_pred,se_pred\n")

    predict_names = []
    predict_feature = []

    for a in predict_input_vec:
        predict_names.append(a[0])
        predict_feature.append(a[1])

    predict_feature = np.array([np.concatenate(sub_array).tolist() for sub_array in predict_feature], dtype="f")  # 如果多个channel将多个channel的feature数组根据channel数目依次拼接。一个channel的所有feature接着下一个channel的所有feature。
    # predict_feature = np.array([np.array(sub_array).T.flatten().tolist() for sub_array in predict_feature])   # 如果多个channel将多个channel的fenture数组根据feature的次序进行拼接。第一个feature的所有channel接着下一个feature的所有channel。

    pred_predict = model.predict(predict_feature)

    pred_multidict = defaultdict(list)
    for names, preds in zip(predict_names, pred_predict):
        pred_multidict[names].append(float(preds))
    
    for cc in pred_multidict.keys():
        pred_mean = np.mean(np.array(pred_multidict[cc]))
        pred_se = np.std(np.array(pred_multidict[cc]))
        predict_rt.write(cc+","+str(pred_mean)+","+str(pred_se)+"\n")

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

    predict_trainfeature = np.array([np.concatenate(sub_array).tolist() for sub_array in predict_trainfeature], dtype="f")  # 如果多个channel将多个channel的feature数组根据channel数目依次拼接。一个channel的所有feature接着下一个channel的所有feature。
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