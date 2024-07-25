import os
import numpy as np
import torch
import json
from collections import Counter
import re
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pickle


def load_data(filename):
    """
    read pickle file data
    :param filename:
    :return:
    """

    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def calculate_model_size(model):
    """
    calculate model size
    :param model:
    :return:
    """

    total_params = 0
    total_size = 0
    for param in model.parameters():
        # Calculation of the number of parameters
        num_params = param.numel()
        total_params += num_params

        # Memory size (in bytes) for calculation parameters
        size = num_params * param.element_size()
        total_size += size

    # Convert to MB
    total_size_mb = total_size / (1024 ** 2)

    return total_params, total_size_mb


def check_path(path):
    """
    check path style
    :param path:
    :return:
    """
    if not path.endswith("/"):
        path += "/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_model(config, model):
    """
    save models
    :param config:
    :param model:
    :return:
    """
    model_saved_path = check_path(config.model_saved_path)
    torch.save(model.state_dict(), model_saved_path + config.model_saved_name)
    print("save model successful！")


def load_model(config, model):
    """
    load model
    :param config:
    :param model:
    :return:
    """
    model.load_state_dict(torch.load(config.model_saved_path + config.model_saved_name))
    print("load model successful!")
    return model


def remove_comments(code):
    """
    remove comments
    :param code:
    :return:
    """

    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return code


def read_c_file(file_name):
    with open(file_name, 'r') as f:
        content = f.read()
    content = remove_comments(content)
    content_list = content.split("\n")
    return content_list


def get_MCM_score(true_labels, predict_labels):
    """
    Calculation of evaluation indicators
    :param true_labels:
    :param predict_labels:
    :return:
    """
    accuracy = accuracy_score(true_labels, predict_labels)
    recall = recall_score(true_labels, predict_labels)
    precision = precision_score(true_labels, predict_labels)
    f1 = f1_score(true_labels, predict_labels)

    return {
        "Precision": format(precision, '.3f'),
        "Recall": format(recall, '.3f'),
        "F1": format(f1, '.3f'),
        "ACC": format(accuracy, '.3f'),
    }


def check_path_file(input_path, file_name):
    """
    Determine if a file exists
    :param input_path:
    :param file_name
    :return:
    """
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    file_path = input_path + file_name
    if os.path.isfile(file_path):
        os.remove(file_path)
        os.mknod(file_path)
    else:
        os.mknod(file_path)


def read_json_files(input_path, file_name):
    """
    Read json file
    :param input_path:
    :param file_name:
    :return:
    """
    with open(input_path + file_name, 'r', encoding="utf-8") as fr:
        lines = fr.readlines()
    return lines


def write_list2txt(output_path, file_name, list_):
    """
    Write list data to txt file
    :param output_path:
    :param file_name:
    :param list_:
    :return:
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(output_path + file_name, 'w', encoding="utf-8") as fw:
        for line in list_:
            fw.write(str(line) + "\n")
