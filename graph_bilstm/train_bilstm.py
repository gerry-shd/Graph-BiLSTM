import torch
import torch.nn as nn
from graph_bilstm.models.BiLSTM import BiLSTM
from graph_bilstm.models.BiLSTM_Attention import BiLSTM_Attention

from torch.utils.data import DataLoader
from graph_bilstm.data_loader import Dataset_Loader_graph, Dataset_Loader_processed
import pandas as pd
import numpy as np
from graph_bilstm import config, utils
from sklearn.model_selection import train_test_split


def generate_dataframe(input_path):
    """
    生成datarame
    :param input_path:
    :param file_name:
    :return:
    """
    input_path = utils.check_path(input_path)
    dic = []
    count_bad = 0
    count_good = 0
    for type_name in os.listdir(input_path):
        dic_name = input_path + type_name
        file_name = glob.glob(dic_name + "/*.pkl")
        for file in file_name:
            if type_name == "good":
                count_good += 1
            if type_name == "bad":
                count_bad += 1
            if count_good > 15000 and type_name == "good":
                continue
            if count_bad > 15000 and type_name == "bad":
                continue
            data_pkl = utils.load_data(file)
            if len(data_pkl[0]) == 0:
                continue
            dic.append({
                "filename": file.split("/")[-1].rstrip(".pkl"),
                "length": len(data_pkl[0]),
                "data": data_pkl,
                "label": 0 if type_name == "good" else 1
            })

    final_dic = pd.DataFrame(dic)
    mean_value = final_dic['length'].mean()
    print("Average Number of Nodes：", mean_value)
    return final_dic


def datas_split(dataset_name):
    """
    split datas
    :return:
    """
    if "graph" in dataset_name:
        data_df = generate_dataframe(config.dataset_source_path)
        vectors = np.array(data_df.iloc[:, 2].values)
        labels = data_df.iloc[:, 3].values
    elif "processed" in dataset_name:
        data_df = pd.read_pickle(config.input_path + dataset_name)
        vectors = np.stack(data_df.iloc[:, 0].values)
        labels = data_df.iloc[:, 1].values
    else:
        raise NotImplementedError

    positive_idxs = np.where(labels == 1)[0]
    negative_idxs = np.where(labels == 0)[0]
    undersampled_negative_idxs = np.random.choice(negative_idxs, len(positive_idxs), replace=True)
    resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])

    X_train, X_test, y_train, y_test = train_test_split(vectors[resampled_idxs,], labels[resampled_idxs],
                                                        train_size=0.8,
                                                        test_size=0.2, stratify=labels[resampled_idxs])
    return X_train, X_test, y_train, y_test


def test(model, test_loader, loss_func):
    model.eval()
    loss_val = 0.0
    predict_labels = []
    true_labels = []
    for data in test_loader:
        datas = data['vectors'].to(config.device)
        labels = data['labels'].to(config.device)
        preds = model(datas)
        loss = loss_func(preds, labels)
        loss_val += loss.item() * datas.size(0)
        preds = torch.argmax(preds, dim=1)
        predict_labels += list(np.array(preds.cpu()))
        true_labels += list(np.array(labels.cpu()))
    test_loss = loss_val / len(test_loader.dataset)
    compute_info = utils.get_MCM_score(true_labels, predict_labels)
    precision = compute_info['Precision']
    recall = compute_info['Recall']
    f1 = compute_info['F1']
    acc = compute_info['ACC']
    print(f"Test: loss: {test_loss}, acc: {acc}, pre:{precision}, recall: {recall}, f1: {f1}")
    return acc, compute_info, true_labels, predict_labels


def train(model, train_loader, test_loader, optimizer, loss_func, epochs):
    best_val_acc = 0.0
    train_losses = []
    test_acces = []
    train_acces = []
    compute_info = None
    for epoch in range(epochs):
        model.train()
        loss_val = 0.0
        predict_labels = []
        true_labels = []
        for data in train_loader:
            datas = data['vectors'].to(config.device)
            labels = data['labels'].to(config.device)
            preds = model(datas)
            loss = loss_func(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val += loss.item() * datas.size(0)
            preds = torch.argmax(preds, dim=1)
            predict_labels += list(np.array(preds.cpu()))
            true_labels += list(np.array(labels.cpu()))
        train_loss = loss_val / len(train_loader.dataset)
        train_losses.append(train_loss)
        compute_info = utils.get_MCM_score(true_labels, predict_labels)
        precision = compute_info['Precision']
        recall = compute_info['Recall']
        f1 = compute_info['F1']
        acc = compute_info['ACC']
        train_acces.append(acc)

        if epoch % 2 == 0:
            print(
                f"{epoch}/{epochs} Train: loss: {train_loss}, acc: {acc}, pre:{precision}, recall: {recall}, f1: {f1}")
            test_acc, test_compute_info, true_labels, predict_labels = test(model, test_loader, loss_func)
            test_acces.append(test_acc)
            if best_val_acc < float(test_acc):
                best_val_acc = float(test_acc)
                utils.save_model(config, model)

    utils.write_list2txt(config.log_saved_path, config.log_train_loss_file, train_losses)
    utils.write_list2txt(config.log_saved_path, config.log_test_acc_file, test_acces)
    utils.write_list2txt(config.log_saved_path, config.log_train_acc_file, train_acces)
    utils.write_list2txt(config.log_saved_path, config.log_test_info_file, [str(test_compute_info)])
    utils.write_list2txt(config.log_saved_path, config.log_train_info_file, [str(compute_info)])


def main():
    if config.dataset_graph:
        dataset_name = config.graph_dataset_name
    else:
        dataset_name = config.processed_dataset_name
    X_train, X_test, y_train, y_test = datas_split(dataset_name)
    print(len(X_train))
    print(len(y_test))
    if "graph" in dataset_name:
        train_set = Dataset_Loader_graph(X_train, y_train, config)
        test_set = Dataset_Loader_graph(X_test, y_test, config)
    elif "processed" in dataset_name:
        train_set = Dataset_Loader_processed(X_train, y_train, config)
        test_set = Dataset_Loader_processed(X_test, y_test, config)
    else:
        raise NotImplementedError

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=True)
    if config.model_name == "BiLSTM":
        model = BiLSTM(config)
    elif config.model_name == "BiLSTM_Attention":
        model = BiLSTM_Attention(config)
    else:
        raise NotImplementedError
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_func = nn.CrossEntropyLoss().to(config.device)
    train(model, train_loader, test_loader, optimizer, loss_func, config.epochs)
    model = utils.load_model(config, model)
    total_params, total_size_mb = utils.calculate_model_size(model)
    test_acc, test_compute_info, true_labels, predict_labels = test(model, test_loader, loss_func)
    utils.write_list2txt(config.log_saved_path, config.log_test_info_file,
                         [str(test_compute_info), f"Total parameters: {total_params}",
                          f'Total size: {total_size_mb:.2f} MB'])
    utils.write_list2txt(config.log_saved_path, config.log_test_predict_label, predict_labels)
    utils.write_list2txt(config.log_saved_path, config.log_test_true_label, true_labels)


if __name__ == '__main__':
    main()
