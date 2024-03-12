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
                                                  name="mean_aggregation_layer_" + str(i + 1)))

        self.dense = tf.keras.layers.Dense(1
                                           , activation=tf.nn.relu
                                           , use_bias=False
                                           , kernel_initializer=init_fn
                                           , name="dense"
                                           )

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        # Using mean squared error loss function.
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 32), dtype=tf.float64),
            tf.TensorSpec(shape=(None, None, 32), dtype=tf.float64),
            tf.TensorSpec(shape=(None, None, None, 32), dtype=tf.float64),
            tf.TensorSpec(shape=(None, 32), dtype=tf.float64),
            tf.TensorSpec(shape=(None, None, 32), dtype=tf.float64),
            tf.TensorSpec(shape=(None, None, None, 32), dtype=tf.float64),
        ])
    def call(self, src, src_neg, src_neg_neg, dst, dst_neg, dst_neg_neg):
        src_neg_aggregated, dst_neg_aggregated = self.seq_layers[1].call(src_neg, src_neg_neg, dst_neg, dst_neg_neg)
        src, dst = self.seq_layers[0].call(src, src_neg_aggregated, dst, dst_neg_aggregated)
        x = tf.concat([src, dst], 1)
        x = self.dense(x)
        return tf.cast(x, tf.float64)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 32), dtype=tf.float64),
            tf.TensorSpec(shape=(None, None, 32), dtype=tf.float64),
            tf.TensorSpec(shape=(None, None, None, 32), dtype=tf.float64),
            tf.TensorSpec(shape=(None, 32), dtype=tf.float64),
            tf.TensorSpec(shape=(None, None, 32), dtype=tf.float64),
            tf.TensorSpec(shape=(None, None, None, 32), dtype=tf.float64),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float64),
        ])
    def train(self, src, src_neg, src_neg_neg, dst, dst_neg, dst_neg_neg, labels):
        with tf.GradientTape() as tape:
            predicted = self.call(src, src_neg, src_neg_neg, dst, dst_neg, dst_neg_neg)
            loss = self.loss_fn(tf.convert_to_tensor(labels), predicted)

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
                                 , dtype=tf.float64
                                 , initializer=init_fn
                                 , trainable=True
                                 )

    def call(self, src, src_neg, dst, dst_neg):
        dst_aggregated = tf.reduce_mean(dst_neg, axis=1, keepdims=False)
        src_aggregated = tf.reduce_mean(src_neg, axis=1, keepdims=False)
        dst_concatenated_features = tf.concat([dst, dst_aggregated], len(dst_aggregated.shape) - 1)
        src_concatenated_features = tf.concat([src, src_aggregated], len(src_aggregated.shape) - 1)
        re_dst = tf.matmul(dst_concatenated_features, self.w)
        re_src = tf.matmul(src_concatenated_features, self.w)
        return re_src, re_dst
