#!/usr/bin/env python3

import tensorflow as tf

# GlorotUniform instantiates an initializer to initializes the parameters of the weight matrix
# where reference from "Understanding the difficulty of training deep feedforward neural networks".
init_fn = tf.keras.initializers.GlorotUniform


class GraphSage(tf.keras.Model):
    def __init__(self, dim, samples, learning_rate):
        """
        Graphsage means graph sample and aggregate, it builds aggregation nodes externally instead of internally.
        :param dim: the input dimension.
        :param samples: the number of aggregation nodes in each layer.
        :param learning_rate: the learning rate.
        """
        super().__init__()

        self.samples = samples

        # Build mean aggregation layers.
        self.seq_layers = []
        for i in range(len(samples)):
            self.seq_layers.append(MeanAggregator(input_dim=dim, output_dim=dim, sample=self.samples[i],
                                                  name="mean_aggregation_layer_" + str(i+1)))

        self.dense_1 = tf.keras.layers.Dense(32
                                             , activation=tf.nn.relu
                                             , use_bias=False
                                             , kernel_initializer=init_fn
                                             , name="dense1"
                                             )
        self.dense_1.build(input_shape=(None, dim * 2))

        self.dense_2 = tf.keras.layers.Dense(2
                                             , activation=tf.nn.softmax
                                             , use_bias=False
                                             , kernel_initializer=init_fn
                                             , name="dense4"
                                             )
        self.dense_2.build(input_shape=(None, 32))

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(64, 1433), dtype=tf.float32),
            tf.TensorSpec(shape=(192, 1433), dtype=tf.float32),
            tf.TensorSpec(shape=(576, 1433), dtype=tf.float32),
            tf.TensorSpec(shape=(64, 1433), dtype=tf.float32),
            tf.TensorSpec(shape=(192, 1433), dtype=tf.float32),
            tf.TensorSpec(shape=(576, 1433), dtype=tf.float32),
        ])
    def call(self, src, src_neg, src_neg_neg, dst, dst_neg, dst_neg_neg):
        src_neg, dst_neg = self.seq_layers[1].call(src_neg, src_neg_neg, dst_neg, dst_neg_neg)
        src, dst = self.seq_layers[0].call(src, src_neg, dst, dst_neg)
        x = tf.concat([src, dst], 1)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(64, 1433), dtype=tf.float32),
            tf.TensorSpec(shape=(192, 1433), dtype=tf.float32),
            tf.TensorSpec(shape=(576, 1433), dtype=tf.float32),
            tf.TensorSpec(shape=(64, 1433), dtype=tf.float32),
            tf.TensorSpec(shape=(192, 1433), dtype=tf.float32),
            tf.TensorSpec(shape=(576, 1433), dtype=tf.float32),
            tf.TensorSpec(shape=(64, 1), dtype=tf.float32),
        ])
    def train(self, src, src_neg, src_neg_neg, dst, dst_neg, dst_neg_neg, inputs_labels):
        with tf.GradientTape() as tape:
            predicted = self.call(src, src_neg, src_neg_neg, dst, dst_neg, dst_neg_neg)
            loss = self.loss_fn(tf.convert_to_tensor(inputs_labels), predicted)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss


################################################################
#                         Custom Layers                        #
################################################################

class MeanAggregator(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, sample, **kwargs):
        """
        Aggregation layer.
        :param input_dim: the input dimension.
        :param output_dim: the output dimension.
        :param sample: the number of aggregation nodes.
        :param kwargs: the layer name.
        """
        super().__init__(**kwargs)
        self.sample = sample
        self.w = self.add_weight(name=kwargs["name"] + "_weight"
                                 , shape=(input_dim * 2, output_dim)
                                 , dtype=tf.float32
                                 , initializer=init_fn
                                 , trainable=True
                                 )

    def call(self, src, src_neg, dst, dst_neg):
        dst_neg = tf.reshape(dst_neg,
                             (int(dst_neg.shape[0] / self.sample), self.sample, dst_neg.shape[1]))
        src_neg = tf.reshape(src_neg,
                             (int(src_neg.shape[0] / self.sample), self.sample, src_neg.shape[1]))

        dst_aggregated = tf.reduce_mean(dst_neg, axis=1)
        src_aggregated = tf.reduce_mean(src_neg, axis=1)

        dst_concatenated_features = tf.concat([dst, dst_aggregated], 1)
        src_concatenated_features = tf.concat([src, src_aggregated], 1)
        re_dst = tf.matmul(dst_concatenated_features, self.w)
        re_src = tf.matmul(src_concatenated_features, self.w)
        return tf.nn.relu(re_src), tf.nn.relu(re_dst)
