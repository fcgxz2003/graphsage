import tensorflow as tf

if __name__ == '__main__':
    import tensorflow as tf

    # 创建一个形状为 [18,1433] 的张量
    tensor = tf.random.normal([18, 5])
    print(tensor)

    # 将张量形状调整为 [6, 3, 1433]
    reshaped_tensor = tf.reshape(tensor, [6, 3, 5])

    # 在第一个维度上进行均值聚合
    aggregated_tensor = tf.reduce_mean(reshaped_tensor, axis=1)

    # 打印结果
    print(aggregated_tensor)