import os
import re

pattern = r"&.*?;|<SUB>.*</SUB>|\\012"


def read_dot_file(input_path, file_name):
    """
    read dot file
    :param input_path:
    :param file_name:
    :return:
    """

    with open(input_path + file_name, 'r', encoding="utf-8") as fr:
        dot = fr.read()
        return dot


def write_dot_file(output_path, file_name, dot):
    """
    write dot file
    :param output_path:
    :param file_name:
    :return:
    """
    if not file_name.endswith("dot"):
        return None
    with open(output_path + file_name, 'w', encoding="utf-8") as fw:
        fw.write(dot)


def dot_clean(input_path, output_path):
    """
    dot clean
    :param input_path:
    :param output_path:
    :return:
    """
    if not input_path.endswith("/"):
        input_path += "/"
    folders = os.listdir(input_path)
    for folder in folders:
        input_path_ = input_path + folder + "/"
        output_path_ = output_path + folder + "/"
        if not os.path.exists(output_path_):
            os.makedirs(output_path_)
        dot_files = os.listdir(input_path_)
        length = len(dot_files)
        count = 0
        for dot_file in dot_files:
            print(f"{input_path_}:{count}/{length}")
            count += 1
            try:
                dot_content = read_dot_file(input_path_, dot_file)
                dot_content = re.sub(pattern, "", dot_content)
                dot_content = dot_content.replace("&amp;", "")
                dot_content = dot_content.replace("&gt;", "")
                dot_content = dot_content.replace("&lt;", "")
                dot_content = dot_content.replace("&quot;", "")
                dot_content = re.sub("<|>", "\"", dot_content)
                dot_content = re.sub("\s-\"", "->", dot_content)
                dot_content = re.sub(" +", " ", dot_content)
                # print(dot_content)
                write_dot_file(output_path_, dot_file, dot_content)
            except Exception as e:
                continue


if __name__ == '__main__':
    dataset_name = "PrimeVul"
    input_path = f"output_step2/{dataset_name}/asts/"
    output_path = f"output_step3/{dataset_name}/clean_asts/"
    dot_clean(input_path, output_path)
    input_path = f"output_step2/{dataset_name}/cpgs/"
    output_path = f"output_step3/{dataset_name}/clean_cpgs/"
    dot_clean(input_path, output_path)
    input_path = f"output_step2/{dataset_name}/pdgs/"
    output_path = f"output_step3/{dataset_name}/clean_pdgs/"
    dot_clean(input_path, output_path)
    input_path = f"output_step2/{dataset_name}/cfgs/"
    output_path = f"output_step3/{dataset_name}/clean_cfgs/"
    dot_clean(input_path, output_path)
