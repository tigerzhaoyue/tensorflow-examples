import math
import tensorflow as tf
import input_cifar
import net_define
import functions
import numpy as np
import os

IMG_W = 32
IMG_H = 32
N_CLASSES = 10
BATCH_SIZE = 32
learning_rate = 0.01
MAX_STEP = 100000


def evaluate():
    with tf.Graph().as_default():
        log_dir = './logs/train/'
        test_dir = './data/cifar-10-batches-bin/'
        n_test = 10000

        images, labels = input_cifar.read_cifar10(data_dir=test_dir,
                                                  is_train=False,
                                                  batch_size = BATCH_SIZE,
                                                  shuffle=False)

        logits = net_define.VGG16(images,N_CLASSES)
        correct = functions.num_correct_prediction(logits, labels)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            print ("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess,ckpt.model_checkpoint_path)
                print ('Loading succesfully, global_step is %s'%global_step)
            else:
                print ('No ckpt files')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord =coord)

            try:
                print('\nEvaluating...')
                num_step = int(math.floor(n_test/BATCH_SIZE))
                num_sample = num_step * BATCH_SIZE
                step = 0
                total_correct = 0
                while step < num_step and not coord.should_stop():
                    batch_correct = sess.run(correct)
                    total_correct += np.sum(batch_correct)
                    step += 1
                print ('total testing samples: %d'%num_sample)
                print ('total correct predictions: %d'%total_correct)
                print ('Average acc: %.2f' %(100 * total_correct/num_sample))

            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

if __name__ =='__main__':
    evaluate()## 88.1%   precision