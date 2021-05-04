@Author : tony
@Date   : 2021/4/29
@Title  : READ_ME about epjb2009 paper practice
@Dec    : READ_ME about 9 measures of similarity and RA LP

代码说明：
practice1.py    :   各算法及AUC评估实现代码（以USAir Dataset为例）
practice2.py    :   各算法及AUC评估实现代码（以Grid Dataset为例）
dataTransform.py:   数据集处理文件（用于处理数据集文件）
topo.py         :   计算图的拓扑结构
data            :   数据集（Grid.txt 符合文中数据集）
data gml        :   包含待处理数据集和处理完成的数据集

API函数说明：
getSimMatrixBy***()    :   各相似性算法实现层
computeBy***()         :   各算法处理层
getAccuracy()          :   依照论文中公式计算AUC
getAccuracyByROC()     :   利用机器学习sklearn中ROC计算AUC
