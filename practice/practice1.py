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

    # for row in data.values:
    #     i = row[1] - 1
    #     j = row[0] - 1
    #     matrix[i][j] = 1
    #     matrix[j][i] = 1

    for row in data.values:
        i = row[1]
        j = row[0]
        matrix[i][j] = 1
        matrix[j][i] = 1

    matrix_df = pd.DataFrame(matrix)
    return matrix_df

# 得到原矩阵的反矩阵
def getContraryMatrix(matrix_df):
    matrix = matrix_df
    matrix1 = matrix.replace({1: 2})
    matrix2 = matrix1.replace({0: 1})
    matrix3 = matrix2.replace({2: 0})
    return matrix3

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
def getAccuracy(multi_grid_train_matrix,multi_grid_nonexist_matrix,data_grid_test):
    count_caseA = 0
    count_caseB = 0
    count_total = 0
    for row in data_grid_test.values:
        # i = row[0] - 1
        # j = row[1] - 1
        i = row[0]
        j = row[1]
        if (multi_grid_train_matrix[i][j] > multi_grid_nonexist_matrix[i][j]):
            count_caseA = count_caseA + 1
        elif (multi_grid_train_matrix[i][j] == multi_grid_nonexist_matrix[i][j]):
            count_caseB = count_caseB + 1
        count_total = count_total + 1
    print(count_total)
    print(count_caseA)
    print(count_caseB)
    AUC = (count_caseA + (0.5 * count_caseB)) / count_total
    return AUC

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

    # CN
    sim_matrix_CN = computeByCN(data_grid_train_adjMatrix)
    AUC_CN = getAccuracy(sim_matrix_CN, data_grid_contrary, data_grid_test)
    print('----AUC CN-----')
    print(AUC_CN)

    # Salton
    sim_matrix_Salton = computeBySalton(data_grid_train_adjMatrix)
    AUC_Salton = getAccuracy(sim_matrix_Salton, data_grid_contrary, data_grid_test)
    print('----AUC_Salton-----')
    print(AUC_Salton)

    # Sorensen
    sim_matrix_Sorensen = computeBySorensen(data_grid_train_adjMatrix)
    AUC_Sorensen = getAccuracy(sim_matrix_Sorensen, data_grid_contrary, data_grid_test)
    print('----AUC_Salton-----')
    print(AUC_Sorensen)

    # HPI
    sim_matrix_HPI = computeByHPI(data_grid_train_adjMatrix)
    AUC_HPI = getAccuracy(sim_matrix_HPI, data_grid_contrary, data_grid_test)
    print('----AUC_HPI-----')
    print(AUC_HPI)

    # HDI
    sim_matrix_HDI = computeByHDI(data_grid_train_adjMatrix)
    AUC_HDI = getAccuracy(sim_matrix_HDI, data_grid_contrary, data_grid_test)
    print('----AUC_HDI-----')
    print(AUC_HDI)

    # LHN
    sim_matrix_LHN = computeByLHN(data_grid_train_adjMatrix)
    AUC_LHN = getAccuracy(sim_matrix_LHN, data_grid_contrary, data_grid_test)
    print('----AUC_LHN-----')
    print(AUC_LHN)

    # PA
    sim_matrix_PA = computeByPA(data_grid_train_adjMatrix)
    AUC_PA = getAccuracy(sim_matrix_PA, data_grid_contrary, data_grid_test)
    print('----AUC_PA-----')
    print(AUC_PA)


    # LP
    sim_matrix_LP = computeByLP(data_grid_train_adjMatrix)
    AUC_LP = getAccuracy(sim_matrix_LP, data_grid_contrary, data_grid_test)
    print('----AUC_LP-----')
    print(AUC_LP)


    print('\n')
    print('\n-------------- Grid DataSet --------------\n')
    auc_CN = Decimal(AUC_CN).quantize(Decimal("0.000"))
    print('AUC CN: %.3f' % auc_CN)

    auc_Salton = Decimal(AUC_Salton).quantize(Decimal("0.000"))
    print('AUC Salton: %.3f' % auc_Salton)

    auc_Sorensen = Decimal(AUC_Sorensen).quantize(Decimal("0.000"))
    print('AUC Sorensen: %.3f' % auc_Sorensen)

    auc_HPI = Decimal(AUC_HPI).quantize(Decimal("0.000"))
    print('AUC HPI: %.3f' % auc_HPI)

    auc_HDI = Decimal(AUC_HDI).quantize(Decimal("0.000"))
    print('AUC HDI: %.3f' % auc_HDI)

    auc_LHN = Decimal(AUC_LHN).quantize(Decimal("0.000"))
    print('AUC LHN: %.3f' % auc_LHN)

    auc_PA = Decimal(AUC_PA).quantize(Decimal("0.000"))
    print('AUC PA: %.3f' % auc_PA)

    auc_LP = Decimal(AUC_LP).quantize(Decimal("0.000"))
    print('AUC LP: %.3f' % auc_LP)

