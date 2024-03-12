#!/usr/bin/env python3
import random
import ipaddress

import numpy as np
import tensorflow as tf
from itertools import islice

from data.cora import load_cora
from graphsage import GraphSage

# The aggregation number of layers.
SAMPLE_SIZES = [3, 3]

# Training parameters
BATCH_SIZE = 64
TRAINING_STEPS = 1000
LEARNING_RATE = 0.001


def generate_training_batch(pairs, num_nodes, batch_size):
    """
    Generate training batch by randomly selecting pairs from train_pair.
    :param pairs: a pair is two nodes connected.
    :param num_nodes: the number of nodes.
    :param batch_size: batch size.
    :return: a batch with positive and negative sample, and it's labels.
    """

    # Control random seed for debug.
    random.seed(10)

    while True:
        # Construct positive sample.
        batch = random.sample(pairs, int(batch_size / 2))
        labels = [random.uniform(0.1, 0.5) for _ in range(int(batch_size / 2))]

        # Construct negative sample.
        while True:
            i = random.randint(0, num_nodes - 1)
            j = random.randint(0, num_nodes - 1)
            if i != j and (i, j) not in pairs:
                batch.append((i, j))
                labels.append(random.uniform(0.5, 1))
            if len(batch) == batch_size:
                break

        # Shuffle batch and labels.
        combined = list(zip(batch, labels))
        random.shuffle(combined)
        batch, labels = zip(*combined)
        yield np.array(batch), np.reshape(np.array(labels), [batch_size, 1])


def select_aggregation_nodes(neighbours, number):
    """
    Select specified number aggregation nodes randomly from neighbours. If the number of neighbor nodes
    is smaller than the specified number, repeated selection so that the input dimension is the same.
    :param neighbours: neighbor nodes.
    :param number: specified aggregation number.
    :return: the index of the aggregation nodes.
    """
    choices = []
    for neighbour in neighbours:
        if len(neighbour) < number:
            choices.append(np.random.choice(neighbour, number, replace=True))
        else:
            choices.append(np.random.choice(neighbour, number, replace=False))
    return np.concatenate(choices)


def generate_aggregation_nodes(pairs, feature, neigh_dict, sample):
    """
    Generate first and second order aggregation nodes by selecting neighbour nodes randomly3
    :param pairs: a pair is two nodes connected.
    :param feature: the feature vector of all nodes.
    :param neigh_dict: the neighbour relationship of all nodes.
    :param sample: the number of aggregation nodes.
    :return: The feature of source nodes and the first, second order neighbor nodes, and
    the feature of destination nodes and the first, second order neighbor nodes.
    """
    # The source and the destination nodes.
    src_node_index = pairs[:, 0]
    dst_node_index = pairs[:, 1]

    # The source and the destination neighbours.
    src_negs = [neigh_dict[index] for index in src_node_index]
    dst_negs = [neigh_dict[index] for index in dst_node_index]

    # Control random seed for debug.
    np.random.seed(13)

    # Select the aggregation nodes in the first order.
    src_layer_1_choices = select_aggregation_nodes(src_negs, sample[0])
    dst_layer_1_choices = select_aggregation_nodes(dst_negs, sample[0])

    # The source and the destination neighbours of first-order neighbor.
    src_layer_1_negs = [neigh_dict[index] for index in src_layer_1_choices]
    dst_layer_1_negs = [neigh_dict[index] for index in dst_layer_1_choices]

    # Select the aggregation nodes in the second order.
    src_layer_2_choices = select_aggregation_nodes(src_layer_1_negs, sample[1])
    dst_layer_2_choices = select_aggregation_nodes(dst_layer_1_negs, sample[1])

    return feature[src_node_index], feature[src_layer_1_choices.reshape((-1, sample[0]))], feature[
        src_layer_2_choices.reshape((-1, sample[0], sample[1]))], \
        feature[dst_node_index], feature[dst_layer_1_choices.reshape(-1, sample[0])], feature[
        dst_layer_2_choices.reshape((-1, sample[0], sample[1]))],


if __name__ == '__main__':
    num_nodes, _, _, _, neigh_dict = load_cora()
    random.seed(10)


    def generate_random_ipv4():
        return '.'.join(str(random.randint(0, 255)) for _ in range(4))


    def ipv4_to_int(ipv4):
        binary_ip = format(int(ipaddress.IPv4Address(ipv4)), '032b')
        feature_vector = [float(bit) for bit in binary_ip]
        return feature_vector


    unique_ips = set()
    # generate ramdom ip address for each node.
    ip_address = np.empty((0, 32), int)
    index = 0
    while len(unique_ips) < num_nodes:
        ip = generate_random_ipv4()
        if ip not in unique_ips:
            unique_ips.add(ip)
            ip_address = np.vstack([ip_address, np.array(ipv4_to_int(ip))])
            index = index + 1

    pair = []
    for i in neigh_dict:
        for j in neigh_dict[i]:
            pair.append((i, j))

    train_pair = pair[:8000]
    test_pair = pair[8000:]

    # Initialize graphsage.
    graphsage = GraphSage(dim=32, samples=SAMPLE_SIZES, learning_rate=LEARNING_RATE)

    # Generate training minibatch.
    batch = generate_training_batch(train_pair, num_nodes, BATCH_SIZE)
    for inputs, labels in islice(batch, 0, TRAINING_STEPS):
        src, src_neg, src_neg_neg, dst, dst_neg, dst_neg_neg = generate_aggregation_nodes(inputs, ip_address,
                                                                                          neigh_dict,
                                                                                          sample=SAMPLE_SIZES)

        loss = graphsage.train(tf.constant(src), tf.constant(src_neg), tf.constant(src_neg_neg),
                               tf.constant(dst), tf.constant(dst_neg), tf.constant(dst_neg_neg),
                               tf.constant(labels))
        print("loss: ", loss)

    # print(graphsage.summary())
    # There is no need to save the "train" function.
    tf.saved_model.save(
        graphsage,
        "model",
        signatures={
            "call": graphsage.call,
            # "train": graphsage.train,
        },
    )
