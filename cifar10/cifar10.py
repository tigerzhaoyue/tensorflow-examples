'''
Cifar-10 demo with tensorflow
You need download cifar10 binary data first in data_dir
Author: Zhao Yue
'''





import tensorflow as tf
import numpy as np
import os
import os.path
import math


BATCH_SIZE = 128
initial_learning_rate = 0.1
MAX_STEP = 200000



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
    image_bytes = img_depth * img_height * img_width #3072


    with tf.name_scope('input'):
        with tf.name_scope('reading_decode'):
            if is_train:
                filename_list = [os.path.join(data_dir,'data_batch_%d.bin' % (num) ) for num in np.arange(1,6)] #1,2,3,4,5
            else:
                filename_list = [os.path.join(data_dir, 'test_batch.bin')]

            filename_queue = tf.train.string_input_producer(filename_list)
            reader = tf.FixedLengthRecordReader(label_bytes + image_bytes) #3073 bytes for one recorder
            key, value = reader.read(filename_queue)
            record_bytes = tf.decode_raw(value, tf.uint8) #according to cifar 10 dataset website documents

        with tf.name_scope('labels'):
            label = tf.slice(record_bytes,[0], [label_bytes])
            label = tf.cast(label, tf.int32)   #for future usage such as tf.nn.in_top_k only receive int32 labels

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
                                    batch_size = batch_size,
                                    num_threads= 16,
                                    capacity = 2000,
                                    min_after_dequeue = 1500)
        else:
            images, label_batch = tf.train.batch(
                                    [image, label],
                                    batch_size = batch_size,
                                    num_threads = 16,
                                    capacity= 2000)

        tf.summary.image('images', images, max_outputs=10)


        ## ONE-HOT while training
        if is_train:
            n_classes = 10
            label_batch_onehot = tf.one_hot(label_batch, depth=n_classes)

            #   shape of(128,) for eval
            return images, tf.reshape(label_batch_onehot, [batch_size, n_classes])
        else:
            return images, tf.reshape(label_batch, [batch_size])


def graph(images):
    '''
    
    :param images: 4D tensor batch*height*width*channel
    :return: 
    '''
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable(name='weights',
                                  shape = [3,3,3,96], #96 filters
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.05, dtype=tf.float32))
        biases = tf.get_variable(name='biases',
                                 shape=[96],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(images,weights,strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope('polliung1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1], strides=[1,2,2,1],padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm1')




    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable(name='weights',
                                  shape=[3, 3, 96, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean = 0.0, stddev=0.05, dtype=tf.float32))
        biases = tf.get_variable(name='biases',
                                 shape=[64],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')


    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                               padding='SAME', name='pooling2')



    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 384],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[384],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)



    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[384, 192],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[192],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)



    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[192, 10],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.004, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[10],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        logits = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')


    return logits




def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        #labels = tf.cast(labels, tf.int64)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        tf.summary.scalar('loss', loss)
    return loss


def train(data_dir,log_dir,ckpt_dir):
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    else:
        tf.gfile.MkDir(log_dir)


    step_counter = tf.Variable(0, name='step',trainable=False)

    images,labels = read_cifar10(data_dir=data_dir,
                                 is_train=True,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True)

    logits = graph(images)
    loss = losses(logits,labels)

    with tf.name_scope('train'):
        # Variables that affect learning rate.
        num_batches_per_epoch = 50000 / BATCH_SIZE
        decay_steps = int(num_batches_per_epoch * 100) #  Epochs after which learning rate decays.(350)

        # Decay the learning rate exponentially based on the number of steps.

        lr = tf.train.exponential_decay(learning_rate=initial_learning_rate,
                                        global_step=step_counter,
                                        decay_steps=decay_steps,
                                        decay_rate=0.5, # Learning rate decay factor.
                                        staircase=True)


        #lr = initial_learning_rate
        tf.summary.scalar('learning_rate', lr)

        # Compute gradients.
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=step_counter)

        saver = tf.train.Saver(tf.global_variables())
        merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            if tf.gfile.Exists(ckpt_dir):
                ckpt = tf.train.get_checkpoint_state(ckpt_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Loading success, global_step is %s' % global_step)
                else:
                    print('No checkpoint file found')

            else:
                sess.run(init)
                print ('No availabel checkpoint_path. init the paras and make the ckpt_dir.....')
                tf.gfile.MkDir(ckpt_dir)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord=coord)
            summary_writer = tf.summary.FileWriter(log_dir,sess.graph)

            try:
                step = step_counter.eval(sess)
                while step < MAX_STEP:
                    if coord.should_stop():
                        break
                    _, loss_value = sess.run([train_op, loss])

                    if step % 10 == 0:
                        print ('Step: %d, loss: %.4f, learning rate: %.4f' % (step, loss_value, sess.run(lr)))

                    if step % 100 == 0:
                        summary_str = sess.run(merged)
                        summary_writer.add_summary(summary_str, step)

                    if step % 500 == 0 or (step + 1) == MAX_STEP:
                        checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)

                    step = step_counter.eval(sess)

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
                coord.request_stop()
            finally:
                coord.request_stop()
                coord.join(threads)




def evaluate(test_dir,log_dir,ckpt_dir):
    with tf.Graph().as_default():
        n_test = 10000

        # reading test data
        images, labels = read_cifar10(data_dir=test_dir,
                                      is_train=False,
                                      batch_size=BATCH_SIZE,
                                      shuffle=False)


        logits = graph(images)
        top_k_op = tf.nn.in_top_k(logits, tf.cast(labels, tf.int32), 1)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                num_iter = int(math.ceil(n_test / BATCH_SIZE))
                true_count = 0
                total_sample_count = num_iter * BATCH_SIZE
                step = 0

                while step < num_iter and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    acc_num_in_this_batch = np.sum(predictions)
                    true_count +=  acc_num_in_this_batch
                    step += 1
                    print('Batch:%d, %d pics accurate in %d' %(step, acc_num_in_this_batch, BATCH_SIZE))
                precision = float(true_count) / float(total_sample_count)
                print('precision = %.3f,true_count = %d,total_sample = %d' % (precision, true_count, total_sample_count))
            except tf.errors.OutOfRangeError: #throw by queuerunner in input_fn
                coord.request_stop()
            finally:
                coord.request_stop()
                coord.join(threads)





if __name__ == '__main__':
    data_dir = '/home/mi/tf/cifar_data'
    log_dir = '/tmp/my_cifar_log'
    ckpt_dir = '/tmp/my_cifar_ckpt'
    test_log_dir = '/tmp/my_cifar_test_log'
    train(data_dir,log_dir,ckpt_dir)
    evaluate(data_dir,test_log_dir,ckpt_dir)
