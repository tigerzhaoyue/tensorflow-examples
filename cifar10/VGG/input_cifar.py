import tensorflow as tf
import numpy as np
import os


def read_cifar10(data_dir, is_train, batch_size, shuffle):
    '''

    :param datadir: the data with cifar 10 binary data
    :param is_train:  bool, is train
    :param batch_size: usually 128
    :param shuffle: if  true shuffle the data to random
    :return: 
        labels 1D tensor tf.int32 shape = [batch_size]
        images:4D tensor tf.float32 shape = [batch_size, height, width, channels]
    '''
    img_width = 32
    img_height = 32
    img_depth = 3
    label_bytes = 1
    image_bytes = img_depth * img_height * img_width  # 3072

    with tf.name_scope('input'):
        with tf.name_scope('reading_decode'):
            if is_train:
                filename_list = [os.path.join(data_dir, 'data_batch_%d.bin' % (num)) for num in
                                 np.arange(1, 6)]  # 1,2,3,4,5
            else:
                filename_list = [os.path.join(data_dir, 'test_batch.bin')]

            filename_queue = tf.train.string_input_producer(filename_list)
            reader = tf.FixedLengthRecordReader(label_bytes + image_bytes)  # 3073 bytes for one recorder
            key, value = reader.read(filename_queue)
            record_bytes = tf.decode_raw(value, tf.uint8)  # according to cifar 10 dataset website documents

        with tf.name_scope('labels'):
            label = tf.slice(record_bytes, [0], [label_bytes])
            label = tf.cast(label, tf.int32)  # for future usage such as tf.nn.in_top_k only receive int32 labels

        with tf.name_scope('images'):
            image_raw = tf.slice(record_bytes, [label_bytes], [image_bytes])
            image_raw = tf.reshape(image_raw, [img_depth, img_height, img_width])
            # from [depth * height * width] to [depth, height, width].
            image = tf.transpose(image_raw, (1, 2, 0))
            image = tf.cast(image, tf.float32)
            image = tf.image.per_image_standardization(image)  # substract off the mean and divide by the variance

        if shuffle:
            images, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=16,
                capacity=2000,
                min_after_dequeue=1500)
        else:
            images, label_batch = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=16,
                capacity=2000)

        tf.summary.image('images', images, max_outputs=10)

        ## ONE-HOT while training
        n_classes = 10
        label_batch = tf.one_hot(label_batch, depth=n_classes)
        label_batch = tf.cast(label_batch, dtype=tf.int32)
        label_batch = tf.reshape(label_batch,[batch_size,n_classes])

        return images, label_batch