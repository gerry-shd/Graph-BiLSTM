import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_path = "datasets/"
dataset_name = "PrimeVul"
dataset_marker = "ast"
dataset_graph = True
# model_name = "BiLSTM"
model_name = "BiLSTM_Attention"
# dataset_source_path = f"{input_path}/{dataset_name}/{dataset_marker}s_pkls"
dataset_source_path = f"/home/gerry/projects/Graph-BiLSTM/code_tools/output_step4_new/PrimeVul/{dataset_marker}s_pkls"

model_saved_path = "model_saved/"
model_saved_name = f"{model_name}_{dataset_name}_{dataset_marker}_model.pkl"
max_node_len = 205

no_vul_flag = "good"
vul_flag = "bad"
output_path = "datasets/"
processed_dataset_name = dataset_name + "_processed.pkl"
graph_dataset_name = dataset_name + "_graph.pkl"
log_saved_path = "log_saved/"
if not dataset_graph:
    log_saved_path = "log_no_graph_saved/"

aggreate = "concat"  # ["mean","sum","concat"]
channels = 1
if aggreate == "concat" and dataset_graph:
    channels = 3

embedding_size = 50
hidden_size = 50
num_layers = 2
lr = 1e-3
batch_size = 64
epochs = 100
is_bidirectional = True
num_class = 2
dropout = 0.3

if not os.path.exists(model_saved_path):
    os.makedirs(model_saved_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)

if not os.path.exists(log_saved_path):
    os.makedirs(log_saved_path)

if dataset_graph:
    log_train_loss_file = f"{model_name}_{dataset_name}_{dataset_marker}_graph_train_loss.txt"
    log_test_acc_file = f"{model_name}_{dataset_name}_{dataset_marker}_graph_test_acc.txt"
    log_train_acc_file = f"{model_name}_{dataset_name}_{dataset_marker}_graph_train_acc.txt"
    log_test_info_file = f"{model_name}_{dataset_name}_{dataset_marker}_graph_test_info.txt"
    log_train_info_file = f"{model_name}_{dataset_name}_{dataset_marker}_graph_train_info.txt"
    log_test_true_label = f"{model_name}_{dataset_name}_{dataset_marker}_graph_test_true_label.txt"
    log_test_predict_label = f"{model_name}_{dataset_name}_{dataset_marker}_graph_test_predict_label.txt"
else:
    log_train_loss_file = model_name + "_" + dataset_name + "_train_loss.txt"
    log_test_acc_file = model_name + "_" + dataset_name + "_test_acc.txt"
    log_train_acc_file = model_name + "_" + dataset_name + "_train_acc.txt"
    log_test_info_file = model_name + "_" + dataset_name + "_test_info.txt"
    log_train_info_file = model_name + "_" + dataset_name + "_train_info.txt"
    log_test_true_label = model_name + "_" + dataset_name + "_test_true_label.txt"
    log_test_predict_label = model_name + "_" + dataset_name + "_test_predict_label.txt"
