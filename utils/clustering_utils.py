# -*- coding:utf-8 -*-
# @Author: yanghao
# @Create time: 2021/08/20 09:11
# @Description: 聚类算法及其评估模块, 包含:
#      class clustering():
#           内置 kmeans(), 自定义距离的 kmeans: kmeans_sd(), 高斯混合模型 GMM(), 层次聚类 Hierarchy(),
#           启发式聚类 spectral(), 密度聚类 DBSCAN(), BIRCH().
#      class evaluate():
#           内置 内部指标评估:internal_evaluate(), 外部指标评估 supervised_evaluate().

from sklearn.cluster import KMeans, AgglomerativeClustering,DBSCAN, SpectralClustering, Birch
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import logging as logger
import numpy as np

class clustering():

    def __init__(self, data: list or np.array, n_cluster: int):
        self.data = data
        self.n_cluster = n_cluster

    def kmeans(self, max_iter: int = 300, n_init: int = 40, num_jobs: int = 1):
        """
        kmeans++ 聚类模型
        :param
            max_iter: 最大迭代次数; 默认为 300.
            n_init: 初始生成采样点个数, 默认为 40.
            num_jobs: 同时生成的任务数量(本函数允许并行运算); 默认为 1, 即只同时进行1个任务. 最大值可取 -1 , 即取最大并行任务数量.
        :return:
            result: 完成聚类的样本的分类指标.
        """
        kmeans_cluster = KMeans(n_clusters = self.n_cluster, max_iter = max_iter,
                                n_init = n_init, init = 'k-means++')
        result = kmeans_cluster.fit_predict(self.data)
        logger.info('*** 聚类结束, 结果已输出 ***')
        return result

    def kmeans_sd(self, distance_function, if_visualize: bool = False):
        """
        基于自定义距离的 kmeans 聚类模型
        *** 在高维空间欧氏距离度量能力变弱的情况下, 可采用其他距离度量方式进行运算 ***
        *** 内置的距离度量方式放入 ../machine_learning/distance_tool.py 文件内 ***
        :param
            distance_function: 自定义的距离函数; 可从 ./distance_tools.py 中调用.
            if_visualize: 是否进行绘图表示聚类结果; 默认为 False. 高维空间下不推荐输出.
        :return:
            clusters: 完成聚类的样本的分类指标.
        """
        initial_centers = kmeans_plusplus_initializer(data = self.data, amount_centers = self.n_cluster).initialize()
        km_metric = distance_metric(type_metric.USER_DEFINED, func = distance_function)
        kmeans_instance = kmeans(self.data, initial_centers, metric = km_metric)
        kmeans_instance.process()
        clusters = kmeans_instance.get_clusters()
        final_centers = kmeans_instance.get_centers()
        if if_visualize:
            kmeans_visualizer.show_clusters(self.data, clusters, final_centers)
        logger.info('*** 聚类结束, 结果已输出 ***')
        return clusters

    def GMM(self, covar_type: str = 'full', to_list: bool = True):
        """
        高斯混合聚类模型
        :param
            covar_type: 组成协方差矩阵的类型. 默认为 'full'; 可选 'spherical', 'diag', 'tied'
            to_list: 是否将聚类结果转化为 list; 默认为 True.
        :return:
            result: 完成聚类的样本的分类指标.
        """
        gmm = GaussianMixture(n_components = self.n_cluster, covariance_type = covar_type, random_state = 2333)
        gmmModel = gmm.fit(self.data)
        result = gmmModel.predict(self.data)
        if to_list:
            result = result.tolist()
        logger.info('*** 聚类结束, 结果已输出 ***')
        return result

    def Hierachy(self, distance_metric: str = 'euclidean', linkage: str = 'average', to_list: bool = False):
        """
        层次聚类
        :param
            distance_metric: 进行距离换算的算法; 格式为 str, 默认为 'euclidean' 欧氏距离; 可选: 'cosine' 余弦相似度,
                            'manhattan' 曼哈顿距离, 'precompute' 不知道是啥.
                            * 当 linkage 为 'ward' 时只能选用 'euclidean'.
            linkage: 联系指标. 默认为 'ward'; 可选: 'complete', 'average', 'single'.
            to_list: 是否将聚类结果转化为列表进行输出; 默认为 True.
        :return:
            result: 完成聚类的样本的分类指标.
        """
        model = AgglomerativeClustering(n_clusters = self.n_cluster, affinity = distance_metric, linkage = linkage)
        result = model.fit_predict(self.data)
        if to_list:
            result = result.tolist()
        logger.info('*** 聚类结束, 结果已输出 ***')
        return result

    def spectral(self, gamma: float = 0.1, to_list: bool = True):
        """
        基于图论的启发式聚类
        :param
            gamma: 核参数; 还不知道有啥用, 先写着. 格式为 float, 默认为 0.1
            to_list: 是否将聚类结果转化为列表进行输出; 默认为 True.
        :return:
            result: 完成聚类的样本的分类指标.
        """
        spectral = SpectralClustering(n_clusters = self.n_cluster, gamma = gamma)
        result = spectral.fit_predict(self.data)
        if to_list:
            result = result.tolist
        return result

    def DBScan(self, epsilon: float = 0.5, min_samples: int = 5, to_list: bool = True):
        """
        基于密度的聚类, 与 kmeans 相似.
        :param
            epsilon: ε-邻域的阈值. 决定了聚类点集边界划分. 当 ε过大时, 容易将不属于当前类的点拉入当前类的判断. 过小可能无法收敛为指定数目.
            min_samples: 每个类的最小含有点的数目; 格式为 int.
            to_list: 是否需要将结果转化为列表; 默认为 True;
        :return:
            result: 完成聚类后的数据标签.
        """
        dbs = DBSCAN(eps = epsilon, min_samples = min_samples)
        result = dbs.fit_predict(self.data)
        if to_list:
            result = result.tolist()
        logger.info('*** 聚类结束, 结果已输出 ***')
        return result

    def BIRCH(self, to_list: bool = True, not_by_k: bool = False):
        """
        中文名全名很长懒得打了
        基于 CF 特征树构建的聚类模型.
        :param
            to_list: 是否需要将结果转化为列表; 默认为 True;
            not_by_k: 不规定聚类数目, 让机器自己迭代出一个它认为ok的聚类情况; 默认为 False.
        :return:
            result: 完成聚类后的数据标签.
        """
        if not_by_k:
            n_cluster = None
        else:
            n_cluster = self.n_cluster
        result = Birch(n_cluster = n_cluster).fit_predict(self.data)
        if to_list:
            result = result.tolist
        logger.info('*** 聚类结束, 结果已输出 ***')
        return result

class model_examiner():

    def __init__(self, true_labels, predict_labels, data):
        self.true = true_labels
        self.predict = predict_labels
        self.data = data

    def internal_evaluate(self, sh_metric: str = 'euclidean'):
        """
        基于两个名字很长的分数执行的聚类用内部评估指标
        第一个很长的: 簇间距离与簇内距离的比值; 理论上越大越好.
        第二个很长的: 轮廓系数; [-1, 1]
        :param
            sh_metric: 计算轮廓系数所使用的距离衡量方法. 默认为 'euclidean' 欧氏距离.
        :return:
            INFO: Calinski-Harabaz and Silhouette scores
        """
        cal_har_score = metrics.calinski_harabasz_score(self.data, self.predict)
        logger.info('Calinski_Harabaz 分数为 % .3f' % cal_har_score)
        silh_score = metrics.silhouette_score(self.data, self.predict, metric = sh_metric)
        logger.info('Silhouette 分数为 % .3f' % silh_score)

    def supervised_evaluate(self):
        """
        基于兰德系数和互信息执行的外部结果评估指标
        * 主要是聚类用 *
        ** 能否用于有监督机器学习模型待检验 **
        :return:
            INFO: adjusted rand_index and adjusted mutual_information
        """
        rand_index = metrics.adjusted_rand_score(self.true, self.predict)
        logger.info('调整兰德系数为 % .3f' % rand_index)
        mutual_info = metrics.adjusted_mutual_info_score(self.true, self.predict)
        logger.info('调整互信息为 % .3f' % mutual_info)
