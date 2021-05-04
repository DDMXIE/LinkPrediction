# @Author : tony
# @Date   : 2021/5/2
# @Title  : epjb2009 paper practice
# @Dec    : deal with the dataset

import networkx as nx

# deal with the INT dataset
def readINT(dataUrl):
    G = nx.read_gml(dataUrl)
    list = dict()
    edge_list = []
    for id, label in enumerate(G.nodes()):
        list[int(label)] = int(id)
    for (v0, v1) in G.edges:
        print(list[int(v0)], list[int(v1)])
        edge_list.append([list[int(v0)], list[int(v1)]])
    return edge_list

# deal with the INT dataset
def readPB(dataUrl):
    G = nx.read_gml(dataUrl)
    list = dict()
    edge_list = []
    for id, label in enumerate(G.nodes()):
        list[label] = int(id)
        print(id, label)
    for (v0, v1) in G.edges:
        print(v0, v1)
        print(list[v0], list[v1])
        edge_list.append([list[v0], list[v1]])
    return edge_list

# deal with the Grid dataset
def readGrid(dataUrl):
    G = nx.read_gml(dataUrl, label='id')
    list = dict()
    edge_list = []
    for id, label in enumerate(G.nodes()):
        list[int(label)] = int(id)
        print(id, label)
    for (v0, v1) in G.edges:
        print(v0, v1)
        print(list[v0], list[v1])
        edge_list.append([list[v0], list[v1]])
    return edge_list

# save the txt
def save(edgeIdList, fileName):
    f = open(fileName, 'w')
    temp = ''
    for item in edgeIdList:
        temp += str(item[0]) + ' ' + str(item[1])
        temp += '\n'
    f.write(temp)
    f.close()

if __name__ == '__main__':
    # print('------------- SRART INT-------------')
    # edge_list = readINT('./data gml/INT.gml')
    # save(edge_list, './data gml/INT.txt')
    # print('------------- INT　END -------------')

    # print('------------- SRART PB-------------')
    # edge_list = readPB('./data gml/PB.gml')
    # print(edge_list)
    # save(edge_list, './data gml/PB.txt')
    # print('------------- PB　END -------------')

    print('------------- SRART Grid-------------')
    edge_list = readGrid('./data gml/Grid.gml')
    print(edge_list)
    save(edge_list, './data gml/Grid.txt')
    print('------------- Grid　END -------------')