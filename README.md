# Hidden code vulnerability detection: a study of the Graph-BiLSTM algorithm

This open-source project is the source code for the paper "Hidden code vulnerability detection: a study of the Graph-BiLSTM algorithm". The objective of this paper is to address the challenge of detecting vulnerabilities in GitHub commit records. This project is primarily organised into three sections, namely code_preprocess, code_tools and graph_bilstm.

## code_preprocess

The principal objective of this tool is to extract function names from GitHub commit records and to extract complete function-level code blocks from the source code based on the function names.

The following open source packages need to be installed:

```shell
Clang install：
    * sudo apt install clang-14
    * sudo apt-get install libclang-dev
    * pip install libclang==14.0.1

Ctags install：
    * sudo apt install ctags
```

## code_tools

This tool mainly computes the graph structure features of the code property graph CPG. Just run it directly according to the STEP steps inside the package.

## graph_bilstm

This tool, which focuses on code vulnerability training and prediction of code vulnerabilities on data features constructed by code_tools.











