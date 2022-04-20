## 2022 泰迪杯数据挖掘竞赛 C 题解题项目说明
### Creater: 今天打羊胎素了吗？ 组
### Create Time: 2022/04/16
### Update Time: 2022/04/20 18:00
#### 1. Test 1 文本无监督分类任务
##### 1.1 数据预处理
待补
##### 1.2 文本向量化
待补
##### 1.3 关键词聚类
######目标：将旅游相关的38个关键字进行聚类分析，获得各聚类的关键字向量的 softmax 描述。
    实验过程中，利用自定义的 clustering_utils 库进行聚类分析
    基于下列函数对向量化后的关键字进行聚类分析
        clustering().kmeans()
        clutering().GMM()
    并利用 Calinski Harabaz 分数及 Silhouette 轮廓系数对聚类效果进行评价。
其部分实验效果如下：

Method: Word2Vec on 2020-2021 title data

|Internal Evaluation|K-means n=3| GMM n=3|K-means n=2|GMM n=2|
|:---:|:---:|:---:|:---:|:---:|
|Calinski Harabaz Score|**194.250**|172.418|168.895|168.895|
|Silhouette Score|0.682|0.705|**0.810**|0.810|

经过多次重复实验，取 Word2Vec 方法的 n=2 的 K-means 聚类算法进行聚类计算
其最终聚类结果如下：

    对应其 38 个关键字
    2019 - Titles
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0]
    2020 - Titles
    [1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    2019 - Texts
    2020 - Texts
    [0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
再利用基于softmax获取类别的表示向量

    利用 utils.utils 库内的 softmax() 函数对其表示向量进行集合训练

##### 1.4 句向量合并
    
##### 1.5 相似度比对