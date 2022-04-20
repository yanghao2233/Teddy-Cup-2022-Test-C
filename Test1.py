# -*- encoding = utf-8 -*-
# @Author:
# @Create Time: 2022/04/07
# @Description:

import jieba
import pandas as pd
import gensim
import logging as logger
import random
# from glove import Glove
# from glove import Corpus
import config
from utils import utils
from utils.clustering_utils import clustering, model_examiner

logger.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s',level = logger.INFO)

def get_keywords(raw_path: str):
    with open(raw_path, 'r', encoding = 'utf-8') as f:
        keywords = [x.strip('\n') for x in f.readlines()]
    return keywords

def preprocesser(raw_path: str, use_bert: bool = False):
    base = pd.read_excel(raw_path, sheet_name = 4)
    titles = base['公众号标题'].astype(str).to_list()
    texts = base['正文'].astype(str).to_list()
    if not use_bert:
        titles = [word_separator(text = x) for x in titles]
        texts = [word_separator(text = x) for x in texts]
    logger.info('Word Separate and Stopwords Delete has been Finished!')
    return titles, texts

def word_separator(text: str = None, cut_all: bool = False, hmm: bool = True, stopwords_path: str = './data/hit_stopwords.txt'):
    """
    文本分词、去停用词函数。
        -> 停用词文本库来源于百度停用词文档。参考 git 仓库为 https://github.com/goto456/stopwords/blob/master/baidu_stopwords.txt
    :param
        text: 需要进行分词的文本，格式为 str。
        cut_all: 剪切的模式，格式为 bool。默认为 False，即精确分词模式。可选 True 模式，为模糊模式。
        hmm: 是否利用 hmm 统计模型进行辅助，格式为 bool。默认为 True。
        stopwords_path: 停用词文本路径， 格式为 str。
    :return:
        result:
    """
    if stopwords_path:
        with open(stopwords_path, 'r', encoding = 'utf-8') as f:
            stopwords = [word.strip('\n') for word in f.readlines()]
    else:
        stopwords = ['NA']
    words = jieba.cut(sentence=text, cut_all=cut_all, HMM=hmm)
    result = [word for word in words if word not in stopwords]
    return result

def get_keyvec(texts: list, model_path: str = None, mode : str = 'd2v'):
    if mode == 'd2v':
        model = gensim.models.doc2vec.Doc2Vec.load(model_path)
        keyvec = [model.docvecs[x] for x in texts]
    else:
        model = gensim.models.word2vec.Word2Vec.load(model_path)
        keyvec = [model.wv[x] for x in texts]
    return keyvec

def keyword_clustering(keyword_path: str, vec_mode: str , model_path: str, n_cluster: int = 2, get_vec: bool = True):
    """
    关键词聚类及聚类后的统一表达获取函数
    :param
        keyword_path: 关键词文件路径。格式为 str。
        vec_mode: 向量文件读取方式，格式为 str。
        model_path: 向量文件路径，格式为 str。
        n_cluster: 聚类数量，格式为 int。
        get_vec: 是否获取表示向量，格式为 bool，默认为 True
    :return:
    """
    with open(keyword_path, 'r', encoding = 'utf-8') as f:
        keywords = [x.strip('\n') for x in f.readlines()]
    if vec_mode == 'd2v':
        keyvecs = [get_keyvec(texts = x, model_path = model_path, mode = vec_mode) for x in [keywords]]
    else:
        keyvecs = get_keyvec(texts = keywords, model_path = model_path, mode = vec_mode)
    cluster_core = clustering(data = keyvecs, n_cluster = n_cluster)
    kmeans = cluster_core.kmeans(max_iter = 300, n_init = 30)
    logger.info('********************* kmeans clustering internal evaluations *********************')
    model_examiner(true_labels = None, predict_labels = kmeans, data = keyvecs).internal_evaluate()
    gaussian = cluster_core.GMM(covar_type = 'full', to_list = True)
    logger.info('********************* GMM clustering internal evaluations *********************')
    model_examiner(true_labels = None, predict_labels = gaussian, data = keyvecs).internal_evaluate()
    print(kmeans.tolist())
    if get_vec: #此处利用 K-means n = 2 with Euclidean distance 进行向量获取
        vecs_0 = [keyvecs[i] for i in range(len(kmeans)) if kmeans[i] == 0]; words_0 = [keywords[i] for i in range(len(kmeans)) if kmeans[i] == 0]
        vecs_1 = [keyvecs[i] for i in range(len(kmeans)) if kmeans[i] == 1]; words_1 = [keywords[i] for i in range(len(kmeans)) if kmeans[i] == 1]
        vecs_0 = utils.softmax_vec_sum(vecs_0); vecs_1 = utils.softmax_vec_sum(vecs_1)
        vecs = [vecs_0, vecs_1]; words = [words_0, words_1]
        return vecs, words

if __name__ == '__main__':
    random.seed(1234)
    raw_path = config.T1_20_all_data_path
    key_path = './data/keywords.txt'
    model_path = config.T1_20_title_model_path
    # titles, texts = preprocesser(raw_path = raw_path)
    # keywords = get_keywords(raw_path = key_path)
    # titles += [keywords]; texts += [keywords]
    # utils.wordvec(texts = titles, window = 6, model_path = model_path)
    key_vecs, key_words = keyword_clustering(keyword_path = key_path, vec_mode = 1,
                                             model_path = model_path, n_cluster = 3, get_vec = True)
    # pd.DataFrame(key_vecs).to_csv(config.T1_20_text_keyvecs_path)