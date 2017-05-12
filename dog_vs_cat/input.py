import tensorflow as tf
import numpy as np
import math
import os



def get_files(file_dir, ratio):
    '''
    
    :param file_dir: file directory
    :param radio: ratio of val sets
    :return: list of images and labels 
    '''

    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print ('There are %d cats and %d dogs.' %(len(cats),len(dogs)))

    image_list = np.hstack((cats,dogs))
    label_list = np.hstack((label_cats,label_dogs))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    all_image_list = temp[:, 0]
    all_label_list = temp[:, 1]

    n_sample = len(all_label_list)
    n_val = math.ceil(n_sample * ratio)  # number of validation samples
    n_train = n_sample - n_val  # number of trainning samples

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    return tra_images, tra_labels, val_images, val_labels


#et_files('./data/test/',0.2)




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

    image = tf.cast(image,tf.string)
    label = tf.cast(label, tf.int32)

    #generate a input queue
    input_queue = tf.train.slice_input_producer([image,label])

    label = input_queue[1]
    image_raw = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_raw, channels=3)


    '''
    if have some data argumentation, insert here
    '''
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

    #barch it with a queue
    image_batch, label_batch = tf.train.batch([image, label],
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
    IMG_W = 208
    IMG_H = 208

    train_dir = './data/train/'
    ratio = 0.2
    tra_images, tra_labels, val_images, val_labels = get_files(train_dir, ratio)
    tra_image_batch, tra_label_batch = get_batch(tra_images, tra_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)



    with tf.Session() as sess:
       i = 0
       coord = tf.train.Coordinator()
       threads = tf.train.start_queue_runners(coord=coord)

       try:
           while not coord.should_stop() and i<1:

               img, label = sess.run([tra_image_batch, tra_label_batch])

               # just test one batch
               for j in np.arange(BATCH_SIZE):
                   print('label: %d' %label[j])
                   plt.imshow(img[j,:,:,:])
                   plt.show()
               i+=1

       except tf.errors.OutOfRangeError:
           print('done!')
           coord.request_stop()
       finally:
           coord.request_stop()
       coord.join(threads)