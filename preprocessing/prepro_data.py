import csv
import networkx as nx


def read_file(filename):
    csvFile = open(filename, "r")
    reader = csv.reader(csvFile)
    result = {}
    for item in reader:
        if 'edges' in filename:
            if int(item[0]) not in result.keys():
                result[int(item[0])] = []
                result[int(item[0])].append(int(item[1]))
            else:
                result[int(item[0])].append(int(item[1]))
        else:
            result[int(item[0])] = int(item[0])
    csvFile.close()
    return result


if __name__ == '__main__':
    edges_dict = read_file("BlogCatalog-dataset/data/edges.csv")
    node_label_dict = read_file("BlogCatalog-dataset/data/group-edges.csv")
    node_dict = read_file("BlogCatalog-dataset/data/nodes.csv")
    group_dict = read_file("BlogCatalog-dataset/data/groups.csv")
    G = nx.Graph()
    for node in node_dict.keys():
        label = node_label_dict[node]
        G.add_node(node, label=label)
    for node in edges_dict.keys():
        ends = edges_dict[node]
        start = node
        for end in ends:
            G.add_edge(start, end)
    nx.write_gexf(G, path="../data/data.gexf")





