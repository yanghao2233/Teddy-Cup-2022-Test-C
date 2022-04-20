
import numpy as np
import scipy.spatial.distance as dist
from scipy.stats import spearmanr
import gensim

def euclide_distance(vec_1, vec_2):
    """

    :param
        vec_1: 需要比对的向量1.
        vec_2: 需要比对的向量2.
    :return:
        score: 该衡量方式的得分.
    """
    distance = np.sqrt(np.sum(np.square(vec_1 - vec_2)))
    return distance

def cosine_similarity(vec_1, vec_2, min1: bool = False):
    """

    :param
        vec_1: 需要比对的向量1.
        vec_2: 需要比对的向量2.
    :return:
        score: 该衡量方式的得分.
    """
    score = np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * (np.linalg.norm(vec_2)))
    if min1:
        score = 1 - score
    return score

def pearson_coefficient(vec_1, vec_2):
    """

    :param
        vec_1: 需要比对的向量1.
        vec_2: 需要比对的向量2.
    :return:
        score: 该衡量方式的额得分.
    """
    score = np.corrcoef(vec_1, vec_2)[1, 0]
    return score

def softmax(vectors):
    vectors -= np.max(vectors)
    return np.exp(vectors) / np.sum(np.exp(vectors))

def mean(vectors):
    return np.mean(vectors)

def wordvec(texts: list = None, window: int = 10, model_path: str = None, if_train: bool = True):
    if if_train:
        model = gensim.models.word2vec.Word2Vec(sentences = texts, vector_size = 128, min_count = 1, window = window, workers = 6)
        if model_path:
            model.save(model_path)
    else:
        model = gensim.models.word2vec.Word2Vec.load(model_path)
    return model

def docvec(texts: list = None, window: int = 5, model_path: str = None, if_train: bool = True):
    if if_train:
        model = gensim.models.doc2vec.Doc2Vec(documents = texts, vector_size = 128, window = window)
        if model_path:
            model.save(model_path)
    else:
        model = gensim.models.doc2vec.Doc2Vec.load(model_path)
    return model

def softmax_vec_sum(vectors: list):
    vectors = softmax(vectors)
    fnl_vec = [0 * len(vectors[1])]
    for vec in vectors:
        fnl_vec += vec
    return fnl_vec

if __name__ == '__main__':
    a = [[1, 2, 3], [2, 3, 4]]
    print(softmax_vec_sum(a))