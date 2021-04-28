# @Author : tony
# @Date   : 2021/4/23
# @Title  : epjb2009 paper practice
# @Dec    : 9 measures of similarity and RA LP

import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from decimal import Decimal
from sklearn import metrics

def loadData():
    data_grid = pd.read_csv('./data/Grid.txt', sep=' ', header=None, index_col=False)
    return data_grid

def initAndMergeData(data_grid):
    # use sklearn to divide the dataset
    train, test, train_label, test_lable = train_test_split(data_grid, range(len(data_grid)), test_size=0.1)

    print('训练集 90% 数据：')
    print(train)
    print('测试集 10% 数据：')
    print(test)

    data_grid_set = (train, test)
    return data_grid_set

def getAdjacencyMatrix(data,len):
    # init the adjacent matrix[i][j]
    matrix = [[0] * len for i in range(len)]

    for row in data.values:
        i = row[1] - 1
        j = row[0] - 1
        matrix[i][j] = 1
        matrix[j][i] = 1

    # for row in data.values:
    #     i = row[1]
    #     j = row[0]
    #     matrix[i][j] = 1
    #     matrix[j][i] = 1

    matrix_df = pd.DataFrame(matrix)
    return matrix_df

# 得到原矩阵的反矩阵
def getContraryMatrix(matrix_df):
    matrix = matrix_df
    matrix1 = matrix.replace({1: 2})
    matrix2 = matrix1.replace({0: 1})
    matrix3 = matrix2.replace({2: 0})
    return matrix3

def getContraryDf(matrix_df):
    rowCount = 0
    matrix_new = []
    for row in matrix_df.values:
        rowArr = np.array(row)
        g = np.where(rowArr == 1)
        itemList = []
        for each in g[0]:
            itemList.append(rowCount)
            itemList.append(each)
            matrix_new.append(itemList)
            itemList = []
        rowCount = rowCount + 1
    matrix_new_df = pd.DataFrame(matrix_new)
    print('-------不存在的边的Dataframe------')
    print(matrix_new_df)
    return matrix_new_df


# CN的相似度矩阵算法
def getSimMatrixByCN(matrix1, matrix2):
    # scipy转变为稀疏矩阵
    a = sp.sparse.csr_matrix(matrix1)
    b = sp.sparse.csr_matrix(matrix2)
    # 矩阵乘法
    matrix = a.dot(b)
    # 转化成dataframe形式
    matrix_df = pd.DataFrame(matrix.todense())
    return matrix_df

# Salton的相似度矩阵算法
def getSimMatrixBySalton(matrix):
    cos_matrix = cosine_similarity(matrix)
    print('----cos_matrix----')
    print(cos_matrix)
    return cos_matrix

# Sorensen的相似度矩阵算法
def getSimMatrixBySorensen(matrix):
    matrix_new = getSimMatrixByCN(matrix, matrix)
    sim_matrix = []
    rowCount = 0
    for row in matrix_new.values:
        rowArr = np.array(row)
        degreeArr = np.array(matrix.sum(axis=0))
        arrSum = degreeArr + rowArr[rowCount]
        arrSum[arrSum == 0] = 1
        item = rowArr / np.array(arrSum)
        sim_matrix.append(item)
        rowCount = rowCount + 1
        print('-----computing-----')
    sim_matrix_df = pd.DataFrame(sim_matrix)
    return sim_matrix_df

# HPI的相似度矩阵算法
def getSimMatrixByHPI(matrix):
    matrix_new = getSimMatrixByCN(matrix, matrix)
    sim_matrix = []
    rowCount = 0
    for row in matrix_new.values:
        arr = np.array(matrix.sum(axis=0))
        arr1 = arr
        arr1[arr1 > arr1[rowCount]] = arr1[rowCount]
        arr1[arr1 == 0] = 1
        c = np.array(row) / np.array(arr1)
        sim_matrix.append(c)
        rowCount = rowCount + 1
        print('-----computing-----')
    sim_matrix_df = pd.DataFrame(sim_matrix)
    return sim_matrix_df

# HDI的相似度矩阵算法
def getSimMatrixByHDI(matrix):
    matrix_new = getSimMatrixByCN(matrix, matrix)
    sim_matrix = []
    rowCount = 0
    for row in matrix_new.values:
        arr = np.array(matrix.sum(axis=0))
        arr1 = arr
        arr1[arr1 < arr1[rowCount]] = arr1[rowCount]
        arr1[arr1 == 0] = 1
        c = np.array(row) / np.array(arr1)
        sim_matrix.append(c)
        rowCount = rowCount + 1
        print('-----computing-----')
    sim_matrix_df = pd.DataFrame(sim_matrix)
    return sim_matrix_df

# LHN的相似度矩阵算法
def getSimMatrixByLHN(matrix):
    matrix_new = getSimMatrixByCN(matrix, matrix)
    sim_matrix = []
    rowCount = 0
    for row in matrix_new.values:
        rowArr = np.array(row)
        degreeArr = np.array(matrix.sum(axis=0))
        item = degreeArr * (rowArr[rowCount])
        item[item == 0] = 1
        c = np.array(row) / np.array(item)
        sim_matrix.append(c)
        rowCount = rowCount + 1
        print('-----computing-----')
    sim_matrix_df = pd.DataFrame(sim_matrix)
    return sim_matrix_df


# PA的相似度矩阵算法
def getSimMatrixByPA(matrix):
    sim_matrix = []
    rowCount = 0
    for row in matrix.values:
        rowArr = np.array(row)
        degreeArr = np.array(matrix.sum(axis=0))
        item = degreeArr * (rowArr[rowCount])
        sim_matrix.append(item)
        rowCount = rowCount + 1
        print('-----computing-----')
    sim_matrix_df = pd.DataFrame(sim_matrix)
    return sim_matrix_df

# LP的相似度矩阵算法
def getSimMatrixByLP(matrix1, matrix2):
    # scipy转变为稀疏矩阵
    a = sp.sparse.csr_matrix(matrix1)
    b = sp.sparse.csr_matrix(matrix2)
    # 矩阵乘法
    matrix = a.dot(b)
    matrix2 = matrix.dot(a)
    # 转化成dataframe形式
    matrix_df_A2 = pd.DataFrame(matrix.todense())
    matrix_df_A3 = pd.DataFrame(matrix2.todense())
    sim_matrix = matrix_df_A2 + (0.001 * matrix_df_A3)
    return sim_matrix


# CN算法处理
def computeByCN(data_grid_train_adjMatrix):
    # 得到train*train的矩阵
    multi_grid_train_matrix_CN = getSimMatrixByCN(data_grid_train_adjMatrix, data_grid_train_adjMatrix)
    print('-------CN simMatrix-------')
    print(multi_grid_train_matrix_CN)

    # 得到data_grid*data_grid的矩阵
    # multi_grid_nonexist_matrix_CN = getSimMatrixByCN(data_grid_adjMatrix, data_grid_adjMatrix)
    # print('-------CN nonexist-------')
    # print(multi_grid_nonexist_matrix_CN)

    return multi_grid_train_matrix_CN

# Salton算法处理
def computeBySalton(data_grid_train_adjMatrix):
    # 得到train*train的矩阵
    multi_grid_train_matrix_Salton = getSimMatrixBySalton(data_grid_train_adjMatrix)
    print('-------Salton simMatrix-------')
    print(multi_grid_train_matrix_Salton)

    return multi_grid_train_matrix_Salton

# Sorensen算法处理
def computeBySorensen(data_grid_train_adjMatrix):
    # 得到train*train的矩阵
    sim_Sorensen = getSimMatrixBySorensen(data_grid_train_adjMatrix)
    multi_grid_train_matrix_Salton = sim_Sorensen * 2
    print('-------Salton simMatrix-------')
    print(multi_grid_train_matrix_Salton)

    return multi_grid_train_matrix_Salton

# HPI算法处理
def computeByHPI(data_grid_train_adjMatrix):
    # 得到train*train的矩阵
    multi_grid_train_matrix_HPI = getSimMatrixByHPI(data_grid_train_adjMatrix)
    print('-------HPI simMatrix-------')
    print(multi_grid_train_matrix_HPI)

    return multi_grid_train_matrix_HPI

# HDI算法处理
def computeByHDI(data_grid_train_adjMatrix):
    # 得到train*train的矩阵
    multi_grid_train_matrix_HDI = getSimMatrixByHDI(data_grid_train_adjMatrix)
    print('-------HDI simMatrix-------')
    print(multi_grid_train_matrix_HDI)

    return multi_grid_train_matrix_HDI

# LHN算法处理
def computeByLHN(data_grid_train_adjMatrix):
    # 得到train*train的矩阵
    multi_grid_train_matrix_LHN = getSimMatrixByLHN(data_grid_train_adjMatrix)
    print('-------LHN simMatrix-------')
    print(multi_grid_train_matrix_LHN)

    return multi_grid_train_matrix_LHN

# PA算法处理
def computeByPA(data_grid_train_adjMatrix):
    # 得到train*train的矩阵
    multi_grid_train_matrix_PA = getSimMatrixByPA(data_grid_train_adjMatrix)
    print('-------PA simMatrix-------')
    print(multi_grid_train_matrix_PA)

    return multi_grid_train_matrix_PA

# LP算法处理
def computeByLP(data_grid_train_adjMatrix):
    # 得到train*train的矩阵
    multi_grid_train_matrix_LP = getSimMatrixByLP(data_grid_train_adjMatrix, data_grid_train_adjMatrix)
    print('-------LP simMatrix-------')
    print(multi_grid_train_matrix_LP)

    return multi_grid_train_matrix_LP


# 计算AUC
def getAccuracy(multi_grid_train_matrix, nonexist_matrix, data_grid_test):
    rowCount = 0
    missingLinkList = []
    nonexistentLinkList = []
    for row in data_grid_test.values:
        i = row[0] - 1
        j = row[1] - 1
        # i = row[0]
        # j = row[1]
        missingLinkList.append(multi_grid_train_matrix[i][j])
        rowCount = rowCount + 1
        print('-----computing1-----')

    rowCount2 = 0
    for row2 in nonexist_matrix.values:
        i = row2[0]
        j = row2[1]
        # i = row[0]
        # j = row[1]
        nonexistentLinkList.append(multi_grid_train_matrix[i][j])
        rowCount2 = rowCount2 + 1
        print('-----computing2-----')

    missingLinkList_np = np.array(missingLinkList)
    nonexistentLinkList_np = np.array(nonexistentLinkList)

    temp = 0    # 分子
    deno = 0    # 分母
    count = 0
    for i in len(missingLinkList_np):
        count_caseA = np.sum(nonexistentLinkList_np < missingLinkList_np[i])
        count_caseB = np.sum(nonexistentLinkList_np == missingLinkList_np[i])
        temp = temp + count_caseA
        temp = temp + (count_caseB * 0.5)
        deno = deno + len(nonexistentLinkList_np)
        count = count + 1

    auc = temp / deno
    print('----auc----')
    print(auc)
    # print('--missingLinkList')
    # missingLinkList_df = pd.DataFrame(missingLinkList)
    # missingLinkList_df[1] = missingLinkList_df[0].rank(ascending=False)
    # print(missingLinkList_df)

    return ''

if __name__ == '__main__':
    # 初始化 解析
    data_grid = loadData()
    print(data_grid)

    # sklearn分割数据集
    data_grid_set = initAndMergeData(data_grid)
    # print(data_grid_set)

    # 获得每个类目的训练集和测试集的数据
    data_grid_train = data_grid_set[0]
    data_grid_test = data_grid_set[1]

    # Dataframe化数据集
    data_grid_df = pd.DataFrame(data_grid)
    data_grid_train_df = pd.DataFrame(data_grid_train)

    # 求出邻接矩阵
    # 总邻接矩阵
    data_grid_adjMatrix = getAdjacencyMatrix(data_grid_df, len(data_grid_df))


    # 训练集的邻接矩阵
    data_grid_train_adjMatrix = getAdjacencyMatrix(data_grid_train,  len(data_grid_df))
    print('-------训练集的邻接矩阵-------')
    print(data_grid_train_adjMatrix)


    # 测试集的邻接矩阵
    data_grid_test_adjMatrix = getAdjacencyMatrix(data_grid_test, len(data_grid_df))
    print('-------测试集的邻接矩阵-------')
    print(data_grid_test_adjMatrix)

    # 求出原邻接矩阵的反矩阵
    print('-------原邻接矩阵-------')
    print(data_grid_adjMatrix)
    data_grid_contrary = getContraryMatrix(data_grid_adjMatrix)
    print('-------原邻接矩阵的反矩阵-------')
    print(data_grid_contrary)

    # 找到不存在的边
    nonexist_matrix = getContraryDf(data_grid_contrary)

    # CN
    sim_matrix_CN = computeByCN(data_grid_train_adjMatrix)
    getAccuracy(sim_matrix_CN, nonexist_matrix, data_grid_test)



