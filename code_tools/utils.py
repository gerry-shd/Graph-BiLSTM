import os
import numpy as np
import json
from collections import Counter
import re


def remove_comments(code):
    """
    Remove comments from c/c++ file
    :param code:
    :return:
    """
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return code


def read_c_file(file_name):
    """
    read a c file
    :param file_name:
    :return:
    """
    with open(file_name, 'r') as f:
        content = f.read()
    # content = remove_comments(content)
    content_list = content.split("\n")
    return content_list


def check_path_file(input_path, file_name):
    """
    Check if the file exists, if it exists remove and rebuild, if it doesn't then create
    :param input_path:
    :param file_name:
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
    Read JSON file
    :param input_path:
    :param file_name:
    :return:
    """
    with open(input_path + file_name, 'r', encoding="utf-8") as fr:
        lines = fr.readlines()
    return lines


def check_dir(path):
    """
    check path
    :param path:
    :return:
    """

    if not os.path.exists(path):
        os.makedirs(path)
    if not path.endswith('/'):
        path = path + '/'
    return path


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


def write_dicts2json(input_path, file_name, dict_):
    """
    Write dicts to json file
    :param input_path:
    :param file_name:
    :param dict_:
    :return:
    """
    with open(input_path + file_name, 'w', encoding="utf-8") as fw:
        fw.write(str(dict_) + "\n")


def write_dict2json(input_path, file_name, dict_):
    """
    Write dict to json file
    :param input_path:
    :param file_name:
    :param dict_:
    :return:
    """
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    dict_json = json.dumps(dict_, ensure_ascii=False)
    with open(input_path + file_name, 'w', encoding="utf-8") as fp:
        fp.write(dict_json)


def read_txt2list(input_path, file_name):
    """
    Read txt file to list
    :param input_path:
    :param file_name:
    :return:
    """
    data_list = []
    with open(input_path + file_name, 'r', encoding="utf-8") as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip("\n")
            data_list.append(line)
    return data_list


def read_json2dict(input_path, file_name):
    """
    Read json file to dict
    :param input_path:
    :param file_name:
    :return:
    """
    with open(input_path + file_name, 'r', encoding="utf-8") as fp:
        lines = fp.read()
        line = lines.replace("nan", "\"nan\"")
        data = eval(line)
    return data
