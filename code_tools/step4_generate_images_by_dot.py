import networkx as nx
import os
import sent2vec
import pickle
import numpy as np

trained_model_path = "datasets/data_model.bin"

sent2vec_model = sent2vec.Sent2vecModel()
sent2vec_model.load_model(trained_model_path)


def graph_extract(dot):
    graph = nx.drawing.nx_pydot.read_dot(dot)
    return graph


def sentence_embedding(sentence):
    emb = sent2vec_model.embed_sentence(sentence)
    return emb[0]


def image_generation(dot):
    # try:
    pdg = graph_extract(dot)
    labels_dict = nx.get_node_attributes(pdg, 'label')
    labels_code = dict()
    for label, all_code in labels_dict.items():
        code = all_code[all_code.index(",") + 1:-2].split('\\n')[0]
        code = code.replace('static void', "void")
        labels_code[label] = code
    degree_cen_dict = nx.degree_centrality(pdg)
    closeness_cen_dict = nx.closeness_centrality(pdg)
    G = nx.DiGraph()
    G.add_nodes_from(pdg.nodes())
    G.add_edges_from(pdg.edges())
    katz_cen_dict = nx.katz_centrality(G)
    # print(degree_cen_dict)
    # print(closeness_cen_dict)
    # print(harmonic_cen_dict)
    # print(katz_cen_dict)
    degree_channel = []
    closeness_channel = []
    katz_channel = []
    for label, code in labels_code.items():
        line_vec = sentence_embedding(code)
        line_vec = np.array(line_vec)
        degree_cen = degree_cen_dict[label]
        degree_channel.append(degree_cen * line_vec)
        closeness_cen = closeness_cen_dict[label]
        closeness_channel.append(closeness_cen * line_vec)
        katz_cen = katz_cen_dict[label]
        katz_channel.append(katz_cen * line_vec)
    return (degree_channel, closeness_channel, katz_channel)


def write_to_pkl(dot, out, existing_files):
    dot_name = dot.split('/')[-1].split('.dot')[0]
    if dot_name in existing_files:
        return None
    else:
        print(dot_name)
        channels = image_generation(dot)
        if channels == None:
            return None
        else:
            (degree_channel, closeness_channel, katz_channel) = channels
            out_pkl = out + dot_name + '.pkl'
            data = [degree_channel, closeness_channel, katz_channel]
            with open(out_pkl, 'wb') as f:
                pickle.dump(data, f)


def main(input_path, output_path):
    if not input_path.endswith("/"):
        input_path += "/"
    if not output_path.endswith("/"):
        output_path += "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    dot_files = os.listdir(input_path)
    length = len(dot_files)
    count = 0
    for dot in dot_files:
        print(f"{count}/{length}")
        count += 1
        out_file = dot.replace(".dot", ".pkl")
        input_dot_file = input_path + dot
        output_pkl_file = output_path + out_file
        try:
            channels = image_generation(input_dot_file)
        except Exception as e:
            print(e)
            continue
        (degree_channel, closeness_channel, katz_channel) = channels
        data = [degree_channel, closeness_channel, katz_channel]
        with open(output_pkl_file, 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    dataset_name = "PrimeVul"
    flag = "good"
    input_path = f"output_step3/{dataset_name}/clean_cpgs/" + flag
    output_path = f"output_step4/{dataset_name}/cpgs_pkls/" + flag
    main(input_path, output_path)
    input_path = f"output_step3/{dataset_name}/clean_asts/" + flag
    output_path = f"output_step4/{dataset_name}/asts_pkls/" + flag
    main(input_path, output_path)
    input_path = f"output_step3/{dataset_name}/clean_pdgs/" + flag
    output_path = f"output_step4/{dataset_name}/pdgs_pkls/" + flag
    main(input_path, output_path)
    input_path = f"output_step3/{dataset_name}/clean_cfgs/" + flag
    output_path = f"output_step4/{dataset_name}/cfgs_pkls/" + flag
    main(input_path, output_path)
