# encoding=utf-8
import os
import re
from code_tools import utils


def merge_dot_files(input_directory, output_file):
    """
    combine dot files
    :param input_directory:
    :param output_file:
    :return:
    """
    if not input_directory.endswith("/"):
        input_directory += "/"
    dot_files = [f for f in os.listdir(input_directory) if f.endswith('.dot')]
    node_id_map = {}
    current_max_id = 0

    def update_node_ids(line):
        nonlocal current_max_id
        node_ids = re.findall(r'\b\d+\b', line)
        for node_id in node_ids:
            if node_id not in node_id_map:
                current_max_id += 1
                node_id_map[node_id] = str(current_max_id)
            line = line.replace(node_id, node_id_map[node_id])
        return line

    with open(output_file, 'w') as outfile:
        outfile.write('digraph G {\n')

        for dot_file in dot_files:
            with open(os.path.join(input_directory, dot_file), 'r') as infile:
                lines = infile.readlines()
                for line in lines:
                    # Skip the graph declaration and closing braces
                    if line.strip() and not line.startswith('digraph') and line.strip() != '}':
                        updated_line = update_node_ids(line)
                        outfile.write(updated_line)

        outfile.write('}\n')


def joern_parse(input_path, output_path):
    """
    joern parse
    :param input_path:
    :param output_path:
    :return:
    """
    if not input_path.endswith("/"):
        input_path += "/"
    if not output_path.endswith("/"):
        output_path += "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_names = os.listdir(input_path)
    length = len(file_names)
    count = 0
    record_txt = os.path.join(output_path, 'parse_result.txt')
    if not os.path.exists(record_txt):
        os.system('touch ' + record_txt)
    fw = open(record_txt, 'a+')
    pwd = os.getcwd() + "/"
    for file in file_names:
        if count > 20000:
            continue
        count += 1
        print(f"Parse:{count}/{length}")
        file_name = input_path + file
        out_file = file.split('/')[-1].split('.')[0] + ".bin"
        out_file_name = output_path + out_file
        if os.path.exists(out_file_name):
            continue
        if os.path.isfile(pwd + out_file_name):
            continue
        # cmd_str = f"joern-parse {pwd + file_name} --language c -o {pwd + out_file_name}"
        cmd_str = f"joern-parse {pwd + file_name}  -o {pwd + out_file_name}"
        print(cmd_str)
        os.system(cmd_str)
        fw.write(out_file + "\n")
    fw.close()


def joern_export(input_path, output_path_cpgs, output_path_asts, output_path_pdgs, output_path_cfgs):
    """
    bin export
    :param input_path:
    :param output_path_cpgs:
    :param output_path_asts:
    :param output_path_pdgs:
    :param output_path_cfgs:
    :return:
    """
    if not input_path.endswith("/"):
        input_path += "/"
    output_path_cpgs = utils.check_dir(output_path_cpgs)
    output_path_asts = utils.check_dir(output_path_asts)
    output_path_pdgs = utils.check_dir(output_path_pdgs)
    output_path_cfgs = utils.check_dir(output_path_cfgs)

    record_txt = os.path.join(output_path_cpgs, "export_result.txt")
    if not os.path.exists(record_txt):
        os.system('touch ' + record_txt)
    fw = open(record_txt, 'a+')
    file_names = os.listdir(input_path)
    length = len(file_names)
    count = 0
    pwd = os.getcwd() + "/"
    for file in file_names:
        count += 1
        print(f"Export:{count}/{length}")
        if not file.endswith(".bin"):
            continue
        file_name = input_path + file
        out_file = file.split('/')[-1].split('.')[0]

        out_file_name_cpg = output_path_cpgs + out_file
        out_file_name_ast = output_path_asts + out_file
        out_file_name_pdg = output_path_pdgs + out_file
        out_file_name_cfg = output_path_cfgs + out_file

        out_file_name_cpg_dot = output_path_cpgs + out_file + ".dot"
        out_file_name_ast_dot = output_path_asts + out_file + ".dot"
        out_file_name_pdg_dot = output_path_pdgs + out_file + ".dot"
        out_file_name_cfg_dot = output_path_cfgs + out_file + ".dot"

        if os.path.exists(out_file_name_cpg):
            continue
        if os.path.exists(pwd + out_file_name_cpg):
            continue
        if os.path.isfile(pwd + out_file_name_cpg):
            continue
        cmd_str_cpg = f"joern-export {pwd + file_name} --repr cpg14 -o {pwd + out_file_name_cpg}"
        cmd_str_ast = f"joern-export {pwd + file_name} --repr ast -o {pwd + out_file_name_ast}"
        cmd_str_cfg = f"joern-export {pwd + file_name} --repr cfg -o {pwd + out_file_name_pdg}"
        cmd_str_pdg = f"joern-export {pwd + file_name} --repr pdg -o {pwd + out_file_name_cfg}"
        os.system(cmd_str_cpg)
        os.system(cmd_str_ast)
        os.system(cmd_str_cfg)
        os.system(cmd_str_pdg)
        merge_dot_files(out_file_name_cpg, out_file_name_cpg_dot)
        merge_dot_files(out_file_name_ast, out_file_name_ast_dot)
        merge_dot_files(out_file_name_pdg, out_file_name_pdg_dot)
        merge_dot_files(out_file_name_cfg, out_file_name_cfg_dot)
        os.system("rm -rf " + pwd + out_file_name_cpg)
        os.system("rm -rf " + pwd + out_file_name_ast)
        os.system("rm -rf " + pwd + out_file_name_pdg)
        os.system("rm -rf " + pwd + out_file_name_cfg)
        fw.write(out_file + "\n")
    fw.close()


if __name__ == '__main__':
    flag = "good"
    dataset_name = "PrimeVul"
    input_path = f"output_step1/{dataset_name}_Normalize/" + flag
    output_path_bin = f"output_step2/{dataset_name}/bin/" + flag
    output_path_cpgs = f"output_step2/{dataset_name}/cpgs/" + flag
    output_path_asts = f"output_step2/{dataset_name}/asts/" + flag
    output_path_pdgs = f"output_step2/{dataset_name}/pdgs/" + flag
    output_path_cfgs = f"output_step2/{dataset_name}/cfgs/" + flag
    joern_parse(input_path, output_path_bin)
    joern_export(output_path_bin, output_path_cpgs, output_path_asts, output_path_pdgs, output_path_cfgs)
