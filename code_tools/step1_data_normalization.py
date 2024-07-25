import os
import re
from code_tools import utils
from code_tools.clean_gadget import clean_gadget


def normalize(path, output_path):
    folderlist = os.listdir(path)
    for folder in folderlist:
        catefolderlist = os.listdir(path + "/" + folder)
        count = 0
        length = len(catefolderlist)
        for catefolder in catefolderlist:
            filepath = path + "/" + folder + "/" + catefolder
            output_filepath = output_path + "/" + folder + "/" + catefolder
            if not os.path.exists(output_path + "/" + folder):
                os.makedirs(output_path + "/" + folder)
            print(folder)
            print("一共" + str(length) + "条文件，正在处理第" + str(count) + "个文件。。。")
            count += 1
            print(catefolder)
            pro_one_file(filepath, output_filepath)


def pro_one_file(filepath, output_filepath):
    source_codes = utils.read_c_file(filepath)
    if len(source_codes) < 20:
        print("文件行数不足，跳过")
        return
    source_codes_str = "\n".join(source_codes)
    source_codes_str = re.sub("/\*(\*(?!/)|[^*])*\*/|//.*", "", source_codes_str)
    node_str_list = source_codes_str.split("\n")
    item_list = []
    for item in node_str_list:
        if "\"\"" in item:
            continue
        if len(item.strip()) == 0:
            continue
        if item.strip().startswith("*"):
            if item.strip().endswith(";"):
                item_list.append(item)
                continue
            else:
                continue
        if item.strip().startswith("/*"):
            if item.strip().endswith(";"):
                item_list.append(item)
                continue
            else:
                continue
        if item.strip().startswith("#"):
            continue
        item_list.append(item)
    source_codes = clean_gadget(item_list)
    source_codes_str = "\n".join(source_codes)
    with open(output_filepath, "w") as file:
        file.write(source_codes_str)


if __name__ == '__main__':
    dataset_name = "PrimeVul"
    input_path = f"datasets/{dataset_name}"
    output_path = f"output_step1/{dataset_name}_Normalize"
    normalize(input_path, output_path)
