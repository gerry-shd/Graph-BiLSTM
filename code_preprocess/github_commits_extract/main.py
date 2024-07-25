from code_preprocess.github_commits_extract import utils
from code_preprocess.github_commits_extract.funcs_extract import get_func_name, extract_function_code
import re
import os

temp_file = "temp/temp.c"
temp_tags_file = "temp/temp_tag.c"


def source_code_clean(source_code):
    source_code_list = source_code.split("\n")
    patch_raw_lines = [x.strip() for x in source_code_list if "#include" not in x and x.strip() != ""]
    return "\n".join(patch_raw_lines)


def github_commit_processed(commit_code, source_code):
    """
    Github commit code processing
    :param commit_code:
    :param source_code:
    :return:
    """
    minus_code_list = []
    commit_code_list = commit_code.split("\n")
    clean_commit_code_list = []
    for item_line in commit_code_list:
        item_line = item_line.strip()
        item_line = re.sub("/\*.*\*/", "", item_line)
        item_line = re.sub("@@.*@@", "@@", item_line)
        if item_line.startswith("+"):
            continue
        if item_line.startswith("/*"):
            continue
        if item_line.startswith("-/*"):
            continue
        if item_line.startswith("- *"):
            continue
        if item_line.startswith("* "):
            continue
        if item_line == "":
            continue
        if item_line.startswith("#"):
            continue
        if item_line.startswith("-"):
            item_line_ = item_line[1:].strip()
            if item_line_.startswith("* "):
                continue
            if item_line_.startswith("/*"):
                continue
            if item_line_ == "":
                continue
            if item_line_ == "{":
                continue
            if item_line_ == "}":
                continue
        clean_commit_code_list.append(item_line)
    clean_commit_code = "\n".join(clean_commit_code_list)
    clean_commit_code_blocks = clean_commit_code.split("@@")
    for code in clean_commit_code_blocks:
        code = code.strip()

        if code == "":
            continue
        code_list = code.split("\n")
        code_ls = []
        for code_l in code_list:
            if code_l.startswith("-"):
                code_l = code_l.replace("-", "").strip()
                code_ls.append(code_l)
        code_lines_idx = utils.find_closest_idx(source_code, code_ls)
        if len(code_lines_idx) == 0:
            continue
        minus_code_list.append([min(code_lines_idx), max(code_lines_idx)])
    if len(minus_code_list) == 0:
        return None
    return minus_code_list


def main(input_path, json_file, output_path, flag):
    """
    Process the spider json file
    :param input_path:
    :param json_file:
    :param output_path:
    :param flag:
    :return:
    """
    if not os.path.exists("temp"):
        os.mkdir("temp")

    output_path = output_path + flag
    output_path = utils.check_path(output_path)
    fr = open(input_path + json_file, 'r', encoding="utf-8")
    datas = fr.readlines()
    length = len(datas)
    count = 0
    for data in datas:
        print(f"{count}/{length}")
        count += 1
        try:
            data = eval(data)
        except:
            continue
        Patch_Url = data["Patch_Url"]
        Code_Info = data["Code_Info"]
        if len(Code_Info) == 0:
            print(Patch_Url)
            print("Code_info is None")
            continue

        vul_file_name = Patch_Url.split("=")[-1][-10:] + "_" + flag + ".c"
        func_codes = []
        for code in Code_Info:
            Patch_File_Name = code['Patch_File_Name']
            if not Patch_File_Name.endswith("c"):
                print("Not a c file")
                continue
            Patch_File_Code = code['Patch_File_Code']
            Vul_File_Raw_Code = code['Vul_File_Raw_Code']
            Vul_File_Raw_Code = source_code_clean(Vul_File_Raw_Code)
            github_commit_code_indexs = github_commit_processed(Patch_File_Code, Vul_File_Raw_Code)
            if github_commit_code_indexs is None:
                print("GitHub's commit is empty")
                continue
            utils.write_temp_file("temp/", "temp.c", Vul_File_Raw_Code)
            Vul_raw_func_names = get_func_name(temp_file, temp_tags_file)
            Vul_File_Raw_Code_list = Vul_File_Raw_Code.split("\n")
            for func_name in Vul_raw_func_names:
                func_code_indexs = extract_function_code(Vul_File_Raw_Code, func_name)
                if func_code_indexs is None:
                    continue
                for item_start, item_end in list(func_code_indexs.values())[0]:
                    for commit_start, commit_end in github_commit_code_indexs:
                        if item_start <= commit_start and item_end >= commit_end:
                            if Vul_File_Raw_Code_list[item_start - 1].strip() == "" or Vul_File_Raw_Code_list[
                                item_start - 1].strip() == "}":
                                func_code = Vul_File_Raw_Code_list[item_start:item_end]
                            else:
                                func_code = Vul_File_Raw_Code_list[item_start - 1:item_end]
                            func_codes.append("\n".join(func_code))
        if len(func_codes) == 0:
            print(Patch_Url)
            print("No suitable code block was found")
            continue
        utils.write_c_files(output_path, vul_file_name, func_codes)
        count += 1


if __name__ == '__main__':
    input_path = "datasets/"
    output = "output/"
    file_name = "ghostscript.json"
    flag = "bad"
    main(input_path, file_name, output, flag)
