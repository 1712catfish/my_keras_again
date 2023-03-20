import tensorflow as tf


def solve_hardware():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print('Running on TPUv3-8')
        batch_size = 256
    except:
        tpu = None
        strategy = tf.distribute.get_strategy()
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print('Running on GPU with mixed precision')
        batch_size = 64

    print('Number of replicas:', strategy.num_replicas_in_sync)
    print('Recommended batch size: %.i' % batch_size)

    return strategy, tpu, batch_size