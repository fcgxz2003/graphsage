#!/usr/bin/env python3
import random

import numpy as np
import tensorflow as tf
import time
from itertools import islice
from sklearn.metrics import f1_score

from dataloader.cora import load_cora
from graphsage import GraphSageSupervised, Layer1MeanAggregator, Layer2MeanAggregator

#### NN parameters
SAMPLE_SIZES = [3, 3]  # implicit number of layers
INTERNAL_DIM = 128
#### training parameters
BATCH_SIZE = 64  # 256
TRAINING_STEPS = 1000  # 100
LEARNING_RATE = 0.001


# 构建 minibatch
# 随机一部分的节点，作为负样本，就是没有边的pair
def generate_training_batch(train_pair, num_nodes, BATCH_SIZE):
    # 控制随机种子，是为了调试方便
    random.seed(10)
    while True:
        batch = random.sample(train_pair, int(BATCH_SIZE / 2))
        labels = [1 for i in range(int(BATCH_SIZE / 2))]
        while True:
            i = random.randint(0, num_nodes-1)
            j = random.randint(0, num_nodes-1)
            if i != j and (i, j) not in train_pair:
                batch.append((i, j))
                labels.append(0)
            if len(batch) == BATCH_SIZE:
                break

        # 将batch, labels打乱顺序
        combined = list(zip(batch, labels))
        random.shuffle(combined)
        batch, labels = zip(*combined)
        yield np.array(batch), np.reshape(np.array(labels), [BATCH_SIZE, 1])


def aggregated_node_select(node_negs, size):
    """
    传入候选邻居节点，该方法从候选邻居节点中选出size个备选，少则补重复。
    """
    choices = []
    for node_neg in node_negs:
        if len(node_neg) < size:
            choices.append(np.random.choice(node_neg, size, replace=True))
        else:
            choices.append(np.random.choice(node_neg, size, replace=False))

    return np.concatenate(choices)


# 默认为[5,5]聚合
def generate_aggregator_node(nodes_index, feature, neigh_dict, sample):
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
    src_layer_1_choices = aggregated_node_select(src_negs, sample[0])
    dst_layer_1_choices = aggregated_node_select(dst_negs, sample[0])
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

    src_layer_2_choices = aggregated_node_select(src_layer_1_negs, sample[1])
    dst_layer_2_choices = aggregated_node_select(dst_layer_1_negs, sample[1])
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

    pair_length = len(pair)  # 10556 , 分8000作为训练，剩下的2556为测试
    train_pair = pair[:8000]
    test_pair = pair[8000:]

    # graphsage 初始化，不将特征传入，而是特征在外部解析。
    graphsage = GraphSageSupervised(dim=len(raw_features[0]), sample=SAMPLE_SIZES, LEARNING_RATE=LEARNING_RATE)

    # 构建 minibatch
    # 随机一部分的节点，作为负样本，就是没有边的pair
    minibatch_generator = generate_training_batch(train_pair, num_nodes, BATCH_SIZE)
    for inputs, inputs_labels in islice(minibatch_generator, 0, TRAINING_STEPS):
        # print("----")
        # print(inputs)
        # print(inputs_labels)
        # src [BATCH_SIZE,1433]
        # src_neg [BATCH_SIZE*3,1433]
        # src_neg_neg [BATCH_SIZE*3*3,1433]
        src, src_neg, src_neg_neg, dst, dst_neg, dst_neg_neg = generate_aggregator_node(inputs, raw_features,
                                                                                        neigh_dict,
                                                                                        sample=SAMPLE_SIZES)

        # print("测试聚合")
        # # 需要转成tf 的格式
        # test = Layer2MeanAggregator(src_neg.shape[-1], src_neg.shape[-1], SAMPLE_SIZES[1], name="test_layer_1")
        # src_neg, dst_neg = test.call(src_neg, src_neg_neg, dst_neg, dst_neg_neg)
        # print(src_neg, dst_neg)
        #
        # test1 = Layer1MeanAggregator(src.shape[-1], src.shape[-1], SAMPLE_SIZES[1], name="test_layer_2")
        # src, dst = test1.call(src, src_neg, dst, dst_neg)
        # print(src, dst)
        #
        # re = graphsage.call(src, src_neg, src_neg_neg, dst, dst_neg, dst_neg_neg)
        # print(re)

        loss = graphsage.train(src, src_neg, src_neg_neg, dst, dst_neg, dst_neg_neg, inputs_labels)
        print("loss: ", loss)

    # optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    #
    # minibatch_generator = generate_training_minibatch(train_nodes, labels, BATCH_SIZE)
    #
    # times = []
    # for inputs, inputs_labels in islice(minibatch_generator, 0, TRAINING_STEPS):
    #     start_time = time.time()
    #     with tf.GradientTape() as tape:
    #         predicted = graphsage(inputs)
    #         loss = loss_fn(tf.convert_to_tensor(inputs_labels), predicted)
    #
    #     grads = tape.gradient(loss, graphsage.trainable_weights)
    #     optimizer.apply_gradients(zip(grads, graphsage.trainable_weights))
    #     end_time = time.time()
    #     times.append(end_time - start_time)
    #     print("Loss:", loss.numpy())
    #
    # # testing
    # results = graphsage(build_batch(test_nodes, neigh_dict, SAMPLE_SIZES))
    # score = f1_score(labels[test_nodes], results.numpy().argmax(axis=1), average="micro")
    # print("Validation F1: ", score)
    # print("Average batch time: ", np.mean(times))
