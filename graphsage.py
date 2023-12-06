#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

init_fn = tf.keras.initializers.GlorotUniform


class GraphSageSupervised(tf.keras.Model):
    def __init__(self, dim, sample, num_classes):
        super().__init__()
        self.layer1 = Layer1MeanAggregator(src_dim=dim, dst_dim=dim, sample=sample[0], name="layer1")
        self.layer2 = Layer2MeanAggregator(src_dim=dim, dst_dim=dim, sample=sample[1], name="layer2")
        self.dense1 = tf.keras.layers.Dense(128
                                            , activation=tf.nn.relu
                                            , use_bias=False
                                            , kernel_initializer=init_fn
                                            , name="dense1"
                                            )

        self.dense2 = tf.keras.layers.Dense(64
                                            , activation=tf.nn.relu
                                            , use_bias=False
                                            , kernel_initializer=init_fn
                                            , name="dense2"
                                            )

        self.dense3 = tf.keras.layers.Dense(8
                                            , activation=tf.nn.relu
                                            , use_bias=False
                                            , kernel_initializer=init_fn
                                            , name="dense3"
                                            )

        self.dense4 = tf.keras.layers.Dense(2
                                            , activation=tf.nn.softmax
                                            , use_bias=False
                                            , kernel_initializer=init_fn
                                            , name="dense4"
                                            )

    def call(self, src, src_neg, src_neg_neg, dst, dst_neg, dst_neg_neg):
        src_neg, dst_neg = self.layer2.call(src_neg, src_neg_neg, dst_neg, dst_neg_neg)
        src, dst = self.layer2.call(src, src_neg, dst, dst_neg)
        x = tf.concat([src, dst], 1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x

    def train(self, src, src_neg, src_neg_neg, dst, dst_neg, dst_neg_neg, inputs_labels):
        with tf.GradientTape() as tape:
            predict = self.call(src, src_neg, src_neg_neg, dst, dst_neg, dst_neg_neg)
            loss = self.compute_uloss(predict, inputs_labels)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss


################################################################
#                         Custom Layers                        #
################################################################

class Layer1MeanAggregator(tf.keras.layers.Layer):
    def __init__(self, src_dim, dst_dim, sample, **kwargs):
        """
        :param int src_dim: input dimension
        :param int dst_dim: output dimension
        """
        super().__init__(**kwargs)
        self.sample = sample  # 记录二阶聚合要聚合的目标数量
        self.w = self.add_weight(name=kwargs["name"] + "_weight"
                                 , shape=(src_dim * 2, dst_dim)
                                 , dtype=tf.float32
                                 , initializer=init_fn
                                 , trainable=True
                                 )

    def call(self, src, src_neg, dst, dst_neg):
        # make (6, 1433) to (2, 3, 1433)
        dst_neg = np.reshape(dst_neg,
                             (int(dst_neg.shape[0] / self.sample), self.sample, dst_neg.shape[-1]))
        src_neg = np.reshape(src_neg,
                             (int(src_neg.shape[0] / self.sample), self.sample, src_neg.shape[-1]))

        # # 构建一个neg_feature 转置矩阵那样维度的，用于计算均值聚合后的点。
        # dif_mat = np.ones([1, neg_feature.shape[0]])
        # dif_mat_sum = np.sum(dif_mat, axis=1, keepdims=True)
        # dif_mat = dif_mat / dif_mat_sum

        # 均值聚合,在第一个维度上进行均值聚合
        dst_aggregated = tf.reduce_mean(dst_neg, axis=1)
        src_aggregated = tf.reduce_mean(src_neg, axis=1)

        dst_concatenated_features = tf.concat([dst, dst_aggregated], 1)
        src_concatenated_features = tf.concat([src, src_aggregated], 1)
        dst = tf.matmul(dst_concatenated_features, self.w)
        src = tf.matmul(src_concatenated_features, self.w)
        return tf.nn.relu(src), tf.nn.relu(dst)


class Layer2MeanAggregator(tf.keras.layers.Layer):
    def __init__(self, src_dim, dst_dim, sample, **kwargs):
        super().__init__(**kwargs)
        self.sample = sample  # 记录二阶聚合要聚合的目标数量
        self.w = self.add_weight(name=kwargs["name"] + "_weight"
                                 , shape=(src_dim * 2, dst_dim)
                                 , dtype=tf.float32
                                 , initializer=init_fn
                                 , trainable=True
                                 )

    # src [2,1433]
    # src_neg [6,1433] 2*3
    # src_neg_neg [18,1433] 2*3*3
    def call(self, src_neg, src_neg_neg, dst_neg, dst_neg_neg):
        # make (18, 1433) to (6, 3, 1433)
        dst_neg_neg = np.reshape(dst_neg_neg,
                                 (int(dst_neg_neg.shape[0] / self.sample), self.sample, dst_neg_neg.shape[-1]))
        src_neg_neg = np.reshape(src_neg_neg,
                                 (int(src_neg_neg.shape[0] / self.sample), self.sample, src_neg_neg.shape[-1]))

        # 均值聚合,在第一个维度上进行均值聚合
        dst_neg_aggregated = tf.reduce_mean(dst_neg_neg, axis=1)
        src_neg_aggregated = tf.reduce_mean(src_neg_neg, axis=1)

        # # 构建一个neg_feature 转置矩阵那样维度的，用于计算均值聚合后的点。 这里后续处理可以加上GAT
        # dif_mat = np.ones([1, 3])
        # dif_mat_sum = np.sum(dif_mat, axis=1, keepdims=True)
        # dif_mat = dif_mat / dif_mat_sum

        dst_neg_concatenated_features = tf.concat([dst_neg, dst_neg_aggregated], 1)
        src_neg_concatenated_features = tf.concat([src_neg, src_neg_aggregated], 1)
        dst_neg = tf.matmul(dst_neg_concatenated_features, self.w)
        src_neg = tf.matmul(src_neg_concatenated_features, self.w)
        return tf.identity(src_neg), tf.identity(dst_neg)
