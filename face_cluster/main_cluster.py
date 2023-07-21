
import numpy as np
from tqdm import tqdm
import infomap
import json
import time
from multiprocessing.dummy import Pool as Threadpool
from multiprocessing import Pool
import multiprocessing as mp
import os
import shutil
import psutil
from tools.utils import Timer, mkdir_if_no_exist, l2norm, intdict2ndarray, read_meta
from evaluation import evaluate, accuracy

import logging
logger = logging.getLogger('main.main_cluster')

class knn():
    def __init__(self, feats, k, index_path='', verbose=True):
        pass

    def filter_by_th(self, i):
        th_nbrs = []
        th_dists = []
        nbrs, dists = self.knns[i]
        for n, dist in zip(nbrs, dists):
            if 1 - dist < self.th:
                continue
            th_nbrs.append(n)
            th_dists.append(dist)
        th_nbrs = np.array(th_nbrs)
        th_dists = np.array(th_dists)
        return th_nbrs, th_dists

    def get_knns(self, th=None):
        if th is None or th <= 0.:
            return self.knns
        # TODO: optimize the filtering process by numpy
        # nproc = mp.cpu_count()
        nproc = 1
        with Timer('filter edges by th {} (CPU={})'.format(th, nproc),
                   self.verbose):
            self.th = th
            self.th_knns = []
            tot = len(self.knns)
            if nproc > 1:
                pool = mp.Pool(nproc)
                th_knns = list(
                    tqdm(pool.imap(self.filter_by_th, range(tot)), total=tot))
                pool.close()
            else:
                th_knns = [self.filter_by_th(i) for i in range(tot)]
            return th_knns


class knn_faiss(knn):
    """
    内积暴力循环
    归一化特征的内积等价于余弦相似度
    """
    def __init__(self,
                 feats,
                 k,
                 index_path='',
                 knn_method='faiss-cpu',
                 verbose=True):
        import faiss
        with Timer('[{}] build index {}'.format(knn_method, k), verbose):
            knn_ofn = index_path + '.npz'
            if os.path.exists(knn_ofn):
                logger.info('[{}] read knns from {}'.format(knn_method, knn_ofn))
                self.knns = np.load(knn_ofn)['data']
            else:
                feats = feats.astype('float32')
                size, dim = feats.shape
                logger.info('feats.shape size: {}, dim: {}'.format(size, dim))
                if knn_method == 'faiss-gpu':
                    import math
                    i = math.ceil(size/1000000)
                    if i > 1:
                        i = (i-1)*4
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(i * 1024 * 1024 * 1024)
                    index = faiss.GpuIndexFlatIP(res, dim)
                else:
                    index = faiss.IndexFlatIP(dim)
                index.add(feats)
        with Timer('[{}] query topk {}'.format(knn_method, k), verbose):
            knn_ofn = index_path + '.npz'
            if os.path.exists(knn_ofn):
                pass
            else:
                sims, nbrs = index.search(feats, k=k) # sims是向量的内积，变换成距离需要做1-sims
                #print("sims:{}, nbrs:{}".format(sims, nbrs))
                # torch.cuda.empty_cache()
                self.knns = [(np.array(nbr, dtype=np.int32),
                              1 - np.array(sim, dtype=np.float32))
                             for nbr, sim in zip(nbrs, sims)]
                #print("self.knns:{}".format(self.knns))


def knns2ordered_nbrs(knns, sort=True):
    if isinstance(knns, list):
        knns = np.array(knns)
    nbrs = knns[:, 0, :].astype(np.int32)
    dists = knns[:, 1, :]
    if sort:
        # sort dists from low to high
        nb_idx = np.argsort(dists, axis=1)
        idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
        dists = dists[idxs, nb_idx]
        nbrs = nbrs[idxs, nb_idx]
    return dists, nbrs


# 构造边
def get_links(single, links, nbrs, dists, args):
    for i in tqdm(range(nbrs.shape[0])):
        count = 0
        for j in range(1, len(nbrs[i])):
            # 排除本身节点
            if i == nbrs[i][j]:
                pass
            elif dists[i][j] <= 1 - args.min_sim:
                # 余弦夹角作为距离需要用 1 - cos_theta
                count += 1
                links[(i, nbrs[i][j])] = float(1 - dists[i][j])
            else:
                break
        # 统计孤立点
        if count == 0:
            single.append(i)
    return single, links


def cluster_by_infomap(nbrs, dists, args):
    """
    基于infomap的聚类
    :param nbrs: 
    :param dists: 
    :param args: 
    :return: 
    """
    single = []
    links = {}
    with Timer('get links', verbose=True):
        single, links = get_links(single=single, links=links, nbrs=nbrs, dists=dists, args=args)

    logger.info('RAM used {}MB'.format(psutil.virtual_memory()[3]/1000000))
    logger.info("start copying links")
    infomapWrapper = infomap.Infomap("--two-level --directed")
    for (i, j), sim in tqdm(links.items()):
        _ = infomapWrapper.addLink(int(i), int(j), sim)

    logger.info('RAM used {}MB'.format(psutil.virtual_memory()[3]/1000000))
    # 聚类运算
    logger.info("start running infomap")
    infomapWrapper.run()

    label2idx = {}
    idx2label = {}

    logger.info('RAM used {}MB'.format(psutil.virtual_memory()[3]/1000000))
    # 聚类结果统计
    for node in infomapWrapper.iterTree():
        # node.physicalId 特征向量的编号
        # node.moduleIndex() 聚类的编号
        idx2label[node.physicalId] = node.moduleIndex()
        if node.moduleIndex() not in label2idx:
            label2idx[node.moduleIndex()] = []
        label2idx[node.moduleIndex()].append(node.physicalId)

    node_count = 0
    for k, v in label2idx.items():
        if k == 0:
            node_count += len(v[2:])
            label2idx[k] = v[2:]
            # logger.info(k, v[0:])
        else:
            node_count += len(v[1:])
            label2idx[k] = v[1:]
            # logger.info(k, v[0:])

    # 孤立点个数
    logger.info("=> Single cluster:{}".format(len(single)))

    keys_len = len(list(label2idx.keys()))
    # logger.info(keys_len)

    # 孤立点放入到结果中
    for single_node in single:
        idx2label[single_node] = keys_len
        label2idx[keys_len] = [single_node]
        keys_len += 1

    logger.info("=> Total clusters:{}".format(keys_len))

    idx_len = len(list(idx2label.keys()))
    logger.info("=> Total nodes:{}".format(idx_len))
    return idx2label, label2idx



def get_dist_nbr(features, args):

    index = knn_faiss(feats=features, k=args.k, knn_method=args.knn_method)
    knns = index.get_knns()
    dists, nbrs = knns2ordered_nbrs(knns)
    #print(dists)
    #print(nbrs)
    return dists, nbrs


def cluster_main(args, extract_features):
    # features = np.fromfile(feature_path, dtype=np.float32)
    features = extract_features.reshape(-1, 256)
    features = l2norm(features)

    #print(features.shape)
    #print(features)
    #print("distance{}".format(np.dot(features[0], features[1])))
    dists, nbrs = get_dist_nbr(features, args=args)
    logger.info('dists.shape:{}, nbrs.shape:{}'.format(dists.shape, nbrs.shape))
    idx2label, label2idx = cluster_by_infomap(nbrs, dists, args)


    # 保存聚类结果
    if eval(args.save_result):
        with open(args.pred_label_path, 'w') as of:
            for idx in range(idx_len):
                of.write(str(idx2label[idx]) + '\n')

    # 评价聚类结果
    if eval(args.is_evaluate) and args.label_path is not None:
        pred_labels = intdict2ndarray(idx2label)
        true_lb2idxs, true_idx2lb = read_meta(args.label_path)
        gt_labels = intdict2ndarray(true_idx2lb)
        for metric in args.metrics:
            evaluate(gt_labels, pred_labels, metric)

    merged_label2idx = label2idx
    if eval(args.merge_cluster):
        # merge similar clusters
        logger.info("Start mergeing clusters...")
        merged_label2idx = {}
        removed_label = set()
        for label1 in label2idx:
            if label1 not in removed_label:
                merged_label2idx[label1] = np.asarray(label2idx[label1])

            cluster_dist_dict = []
            for label2 in label2idx:
                if label1 >= label2:
                    continue
                cluster_center1 = features[label2idx[label1]].mean(axis=0)
                cluster_center2 = features[label2idx[label2]].mean(axis=0)
                cluster_dist = 1 - np.dot(cluster_center1, cluster_center2)
                cluster_dist_dict.append((label2, cluster_dist))
                if cluster_dist < args.cluster_dist_thresh:
                    # logger.info('cluster1 {}, cluster2 {}, distance:{}'.format(label1, label2, cluster_dist))
                    if label1 not in removed_label and label2 not in removed_label:
                        merged_label2idx[label1] = np.append(merged_label2idx[label1], label2idx[label2])
                        removed_label.add(label2)
            cluster_dist_dict.sort(key=lambda x:x[1])
            logger.info("cluster {} neighbours: {}".format(label1, cluster_dist_dict))

        node_count = 0
        for label in merged_label2idx:
            #logger.info("label {}, idx {}".format(label, merged_label2idx[label]))
            node_count += len(merged_label2idx[label])
        logger.info("size of merged_label2idx: {}, node_count {}".format(len(merged_label2idx), node_count))


    # 归档图片
    if args.output_picture_path is not None:
        logger.info("=> Start copy pictures to the output path {} ......".format(args.output_picture_path))
        with open('data/tmp/pic_path', 'r') as f:
            content = f.read()
            picture_path_dict = json.loads(content)
        mkdir_if_no_exist(args.output_picture_path)
        shutil.rmtree(args.output_picture_path)
        os.mkdir(args.output_picture_path)
        tmp_pth = 'data/input_pictures/alldata'
        for label, idxs in merged_label2idx.items():
            picture_reuslt_path = args.output_picture_path + '/' + str(label)
            mkdir_if_no_exist(picture_reuslt_path)
            for idx in idxs:
                picture_path = picture_path_dict[str(idx)]
                shutil.copy(picture_path, picture_reuslt_path)
                shutil.copy(picture_path, tmp_pth)
