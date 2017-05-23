import tensorflow as tf
import numpy as np
import math
import os


def get_files(pos_file_dir,neg_file_dir, ratio):
    '''

    :param file_dir: file directory
    :param radio: ratio of val sets
    :return: list of images and labels 
    '''

    pos = []
    label_pos = []
    neg = []
    label_neg = []
    for file in os.listdir(pos_file_dir):
        pos.append(pos_file_dir + file)
        label_pos.append(1)

    for file in os.listdir(neg_file_dir):
        neg.append(neg_file_dir + file)
        label_neg.append(0)

    print ('There are %d POS and %d NEG.' % (len(pos), len(neg)))

    image_list = np.hstack((pos, neg))
    label_list = np.hstack((label_pos, label_neg))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    np.random.shuffle(temp)
    np.random.shuffle(temp)

    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]

    n_sample = len(all_label_list)
    n_val = math.ceil(n_sample * ratio)  # number of validation samples
    n_train = n_sample - n_val  # number of trainning samples

    tra_images = all_image_list[0:int(n_train)]
    tra_labels = all_label_list[0:int(n_train)]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[int(n_train):-1]
    val_labels = all_label_list[int(n_train):-1]
    val_labels = [int(float(i)) for i in val_labels]

    return tra_images, tra_labels, val_images, val_labels


# et_files('./data/test/',0.2)




def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''

    :param image: image list
    :param label: label list
    :param image_W: width
    :param image_H: height
    :param batch_size: batch size
    :param capacity: queue capacity
    :return: 
        image_batch: 4D tensor [batch_size, width, height, channel]
        label_batch: 1D tensor [batch_size], dtype = tf.int32
    '''

    img_list = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # generate a input queue
    input_queue = tf.train.slice_input_producer([img_list, label])

    label = input_queue[1]
    #reader = tf.WholeFileReader()
    #key, value = reader.read(tf.cast(input_queue[0],tf.string))
    value = tf.read_file(input_queue[0])
    img = tf.image.decode_jpeg(value, channels=3)
    #img = tf.image.hsv_to_rgb(tf.cast(img,tf.float32))
    '''
    if have some data argumentation, insert here
    '''
    img = tf.image.resize_images(img, [image_W, image_H],method=tf.image.ResizeMethod.AREA)
    #img = tf.image.resize_image_with_crop_or_pad(img, image_W, image_H)
    #img = tf.image.per_image_standardization(img)

    # barch it with a queue
    image_batch, label_batch = tf.train.batch([img, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    BATCH_SIZE = 4
    CAPACITY = 256
    IMG_W = 227
    IMG_H = 227

    train_pos_dir = '../data_2/train_pos/'
    train_neg_dir = '../data_2/train_neg/'
    ratio = 0.2
    tra_images, tra_labels, val_images, val_labels = get_files(train_pos_dir, train_neg_dir, ratio)
    tra_image_batch, tra_label_batch = get_batch(tra_images, tra_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop() and i < 2:

                img, label = sess.run([tra_image_batch, tra_label_batch])

                # just test one batch
                for j in np.arange(BATCH_SIZE):
                    print('label: %d' % label[j])
                    print (img[j, :, :, :])
                    plt.imshow(img[j, :, :, :])
                    plt.show()
                i += 1

        except tf.errors.OutOfRangeError:
            print('done!')
            coord.request_stop()
        finally:
            coord.request_stop()
        coord.join(threads)