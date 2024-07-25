import json
import os
import shutil
import re
import copy


def calculate_distances(value, value_list):
    distances = []
    for item in value_list:
        distance = abs(value - item)  # 计算值与列表中每个元素的距离
        distances.append(distance)
    return sum(distances)


def find_closest_idx(source_code, substrs):
    """
    找到所有子串的编号
    :param source_code:
    :param substrs:
    :return:
    """
    lines = source_code.split('\n')
    substrs_idx = dict()
    for substr in substrs:
        substrs_idx[substr] = []
    for i, line in enumerate(lines, start=1):
        for substr in substrs:
            if substr in line:
                substrs_idx[substr].append(i)
    substrs_idx_all = substrs_idx.values()
    more_idx = []
    single_idx = []
    for idxs in substrs_idx_all:
        if len(idxs) > 1:
            more_idx.append(idxs)
        else:
            single_idx.extend(idxs)
    final_index = []
    if len(more_idx) != 0:
        for item in more_idx:
            item_dis = [calculate_distances(x, single_idx) for x in item]
            item_min_idx = item_dis.index(min(item_dis))
            value = item[item_min_idx]
            final_index.append(value)
    final_index.extend(single_idx)
    return final_index


def jaccard_similarity(s1, s2):
    set1 = set(s1)
    set2 = set(s2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0


def find_max_sim_index(given_word, word_list):
    """
    从list中找出与item最相似的单词的索引
    :param item:
    :param word_list:
    :return:
    """
    for word in word_list:
        if given_word in word and not word.endswith(";"):
            return word_list.index(word)
    similarities = [jaccard_similarity(given_word, word) for word in word_list]
    new_sim = copy.deepcopy(similarities)
    new_sim = sorted(new_sim, reverse=True)
    for sim in new_sim:
        index = similarities.index(sim)
        if not word_list[index].endswith(";"):
            return index
    if len(similarities) == 0:
        return None
    # 找到最大的相似度
    max_similarity = max(similarities)

    # 找到最相似的元素的索引
    most_similar_index = similarities.index(max_similarity)

    return most_similar_index


def is_all_uppercase(input_string):
    return input_string.isupper()


def is_all_lowercase(input_string):
    return input_string.islower()


def read_json(input_path, file_name):
    """
    读取单个JSON文件
    :param input_path:
    :param file_name:
    :return:
    """
    with open(input_path + file_name) as f:
        # 读取 JSON 数据
        data = json.load(f)
    return data


def read_jsons(input_path, file_name):
    """
    读取多个JSON格式文件记录
    :param input_path:
    :param file_name:
    :return:
    """
    datas = []
    with open(input_path + file_name) as file:
        # 逐行读取文件内容
        for line in file:
            # 解析每行的 JSON 数据
            data = json.loads(line)
            datas.append(data)
    return datas


def check_path_file(input_path, file_name):
    """
    检查文件是否存在，若存在去除重建，不存在建立
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


def check_path(input_path):
    """
    检查文件是否存在，若存在去除重建，不存在建立
    :param input_path:
    :return:
    """
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    if not input_path.endswith("/"):
        input_path += "/"
    return input_path


def copy_file(src_file, dest_folder):
    # 使用 shutil.copy2() 函数复制文件到目标文件夹
    shutil.copy2(src_file, dest_folder)


def find_key(data, target_key):
    # 如果当前层级为字典类型，则遍历键值对
    if isinstance(data, dict):
        for key, value in data.items():
            # 如果找到目标键，返回对应的值
            if key == target_key:
                return value
            # 如果当前值是嵌套的字典或列表，则继续递归搜索
            elif isinstance(value, (dict, list)):
                result = find_key(value, target_key)
                if result is not None:
                    return result
    # 如果当前层级为列表类型，则遍历列表中的元素
    elif isinstance(data, list):
        for item in data:
            # 如果列表元素是嵌套的字典或列表，则继续递归搜索
            if isinstance(item, (dict, list)):
                result = find_key(item, target_key)
                if result is not None:
                    return result
    # 没有找到目标键，返回None
    return None


def find_values(data, target_key):
    values = []

    # 如果当前层级为字典类型，则遍历键值对
    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                values.append(value)  # 将值添加到列表中
            # 如果当前值是嵌套的字典或列表，则继续递归搜索
            elif isinstance(value, (dict, list)):
                values.extend(find_values(value, target_key))  # 递归调用，并将结果合并到当前列表

    # 如果当前层级为列表类型，则遍历列表中的元素
    elif isinstance(data, list):
        for item in data:
            # 如果列表元素是嵌套的字典或列表，则继续递归搜索
            if isinstance(item, (dict, list)):
                values.extend(find_values(item, target_key))  # 递归调用，并将结果合并到当前列表

    return values


def read_c_file(input_file):
    """
    读取c文件
    :param input_file:
    :return:
    """
    with open(input_file, 'r') as f:
        data_c = f.read()
    return data_c


def write_c_file(output_path, file_name, source_code):
    """
    读取c文件
    :param input_file:
    :return:
    """
    output_path = check_path(output_path)
    with open(output_path + file_name, 'w') as f:
        f.write(source_code)
        # print("源代码写入完成！")


def write_temp_file(output_path, file_name, source_code):
    """
    读取c文件
    :param input_file:
    :return:
    """
    output_path = check_path(output_path)
    with open(output_path + file_name, 'w') as f:
        f.write(source_code)
        print("正在写入临时文件！")


def write_c_files(output_path, file_name, code_list):
    """
    读取c文件
    :param input_file:
    :return:
    """
    output_path = check_path(output_path)
    with open(output_path + file_name, 'w') as f:
        for code in code_list:
            if code is None:
                continue
            f.write(code + "\n")
        print("代码块写入完成！")


def read_c_file_block(input_file, startLine, endLine):
    """
    读取c文件
    :param input_file:
    :param startLine:
    :param endLine:
    :return:
    """
    with open(input_file, 'r') as f:
        data_c = f.readlines()
        data_block = data_c[startLine:endLine + 1]
    return data_block


def find_clause_offsets(full_sentence, sub_clause):
    start_offset = full_sentence.find(sub_clause)
    end_offset = start_offset + len(sub_clause)

    return start_offset, end_offset


def find_line_number(text, offset):
    lines = text.split("\n")
    line_offset = 0
    for i, line in enumerate(lines):
        line_length = len(line) + 1  # 加一是为了包含换行符
        if offset < line_offset + line_length:
            return i + 1  # 行号从1开始计数
        line_offset += line_length

    return -1  # 没有找到对应的行号


def main():
    word = "memcpy((d -> buffer + d -> good),buf,buf_size);"
    source_code = read_c_file("datasets_demo/aviobuf.c")
    start_offset, end_offset = find_clause_offsets(source_code, word)
    print(start_offset)
    print(end_offset)
