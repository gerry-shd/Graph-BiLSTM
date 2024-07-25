import subprocess
import clang.cindex
from code_preprocess.github_commits_extract import utils


def get_func_name(input_file, temp_file):
    cmd = "ctags -R --fields=+n -f " + temp_file + " " + input_file
    subprocess.getoutput(cmd)
    temp_content = utils.read_c_file(temp_file)
    temp_list = temp_content.split("\n")
    func_names = [func.split("\t")[0] for func in temp_list if "(" in func]
    return func_names


def extract_function_code(source_code, function_name):
    # create clang index
    index = clang.cindex.Index.create()
    # Parsing code
    tu = index.parse('temp.c', unsaved_files=[('temp.c', source_code)])
    item_dict = {function_name: []}
    # Traversing the AST
    for node in tu.cursor.walk_preorder():
        if node.spelling == function_name and node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            start_offset = node.extent.start.offset
            end_offset = node.extent.end.offset
            startLine = utils.find_line_number(source_code, start_offset)
            endLine = utils.find_line_number(source_code, end_offset)
            if [startLine, endLine] not in item_dict[function_name]:
                item_dict[function_name].append([startLine, endLine])
    if len(item_dict[function_name]) != 0:
        return item_dict
    return None

# if __name__ == '__main__':
#     input_file = "datasets/temo1.c"
#     output = "temp/temp.c"
#     source_code = utils.read_c_file(input_file)
#     func_names = get_func_name(input_file, output)
#     print(func_names)
#     count = 0
#     for func in func_names:
#         func_code = extract_function_code(source_code, func)
#         if func_code is None:
#             continue
#         count += 1
#         print(func)
#         print(func_code)
#         print("=" * 10)
#     print(len(func_names))
#     print(count)
#     print(func_names)
