#!/usr/bin/env python3
import random

import numpy as np
from itertools import islice

from dataloader.cora import load_cora
from graphsage import GraphSage

# The aggregation number of layers.
SAMPLE_SIZES = [3, 3]
INTERNAL_DIM = 128

# Training parameters
BATCH_SIZE = 64
TRAINING_STEPS = 1000
LEARNING_RATE = 0.001


# 随机一部分的节点，作为负样本，就是没有边的pair
def generate_training_batch(train_pair, num_nodes, batch_size):
    """
    Generate training batch by randomly selecting pairs from train_pair.
    :param train_pair:
    :param num_nodes:
    :param batch_size:
    :return:
    """

    # Control random seed for debug.
    random.seed(10)

    while True:
        # Construct positive sample.
        batch = random.sample(train_pair, int(batch_size / 2))
        labels = [1 for _ in range(int(batch_size / 2))]

        # Construct negative sample.
        while True:
            i = random.randint(0, num_nodes - 1)
            j = random.randint(0, num_nodes - 1)
            if i != j and (i, j) not in train_pair:
                batch.append((i, j))
                labels.append(0)
            if len(batch) == batch_size:
                break

        # Shuffle batch and labels.
        combined = list(zip(batch, labels))
        random.shuffle(combined)
        batch, labels = zip(*combined)
        yield np.array(batch), np.reshape(np.array(labels), [batch_size, 1])


def select_aggregation_nodes(node_negs, size):
    """
    Select aggregation nodes, 传入候选邻居节点，该方法从候选邻居节点中选出size个备选，少则补重复。
    :param node_negs:
    :param size:
    :return:
    """
    choices = []
    for node_neg in node_negs:
        if len(node_neg) < size:
            choices.append(np.random.choice(node_neg, size, replace=True))
        else:
            choices.append(np.random.choice(node_neg, size, replace=False))

    return np.concatenate(choices)


def generate_aggregator_node(nodes_index, feature, neigh_dict, sample):
    """

    :param nodes_index:
    :param feature:
    :param neigh_dict:
    :param sample:
    :return:
    """
    src_node_index = nodes_index[:, 0]
    dst_node_index = nodes_index[:, 1]
    # print("要聚合的src节点和dst节点")
    # print(src_node_index)
    # print(dst_node_index)

    src_negs = [neigh_dict[index] for index in src_node_index]
    dst_negs = [neigh_dict[index] for index in dst_node_index]
    # print("对应的邻居节点下标")
    # print(src_negs)
    # print(dst_negs)

    # 控制随机种子，是为了调试方便
    np.random.seed(13)
    # 如果不足的话，直接随机补全，控制维度相同，否则不好控制输入参数的维度
    src_layer_1_choices = select_aggregation_nodes(src_negs, sample[0])
    dst_layer_1_choices = select_aggregation_nodes(dst_negs, sample[0])
    # print("一阶要聚合的点")
    # print(src_layer_1_choices)
    # print(dst_layer_1_choices)

    # 计算二阶
    # 先把邻居的邻居给找出来
    src_layer_1_negs = [neigh_dict[index] for index in src_layer_1_choices]
    dst_layer_1_negs = [neigh_dict[index] for index in dst_layer_1_choices]
    # print("对应二阶邻居节点的下标")
    # print(src_layer_1_negs)
    # print(dst_layer_1_negs)

    src_layer_2_choices = select_aggregation_nodes(src_layer_1_negs, sample[1])
    dst_layer_2_choices = select_aggregation_nodes(dst_layer_1_negs, sample[1])
    # print("二阶要聚合的点")
    # print(src_layer_2_choices)
    # print(dst_layer_2_choices)

    # 返回src节点特征向量，src邻居节点的特征向量，src邻居的邻居的节点特征向量和
    # dst节点特征向量，dst邻居节点的特征向量，dst邻居的邻居的节点特征向量
    return feature[src_node_index], feature[src_layer_1_choices], feature[src_layer_2_choices], \
        feature[dst_node_index], feature[dst_layer_1_choices], feature[dst_layer_2_choices],


if __name__ == "__main__":
    num_nodes, raw_features, _, _, neigh_dict = load_cora()

    pair = []
    for i in neigh_dict:
        for j in neigh_dict[i]:
            pair.append((i, j))

    train_pair = pair[:8000]
    test_pair = pair[8000:]

    # Initialize graphsage.
    graphsage = GraphSage(dim=len(raw_features[0]), samples=SAMPLE_SIZES, learning_rate=LEARNING_RATE)

    # generate training minibatch
    batch = generate_training_batch(train_pair, num_nodes, BATCH_SIZE)
    for inputs, inputs_labels in islice(batch, 0, TRAINING_STEPS):
        src, src_neg, src_neg_neg, dst, dst_neg, dst_neg_neg = generate_aggregator_node(inputs, raw_features,
                                                                                        neigh_dict,
                                                                                        sample=SAMPLE_SIZES)

        loss = graphsage.train(src, src_neg, src_neg_neg, dst, dst_neg, dst_neg_neg, inputs_labels)
        print("loss: ", loss)
