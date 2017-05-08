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
IS_PRETRAIN = True

def train():
    pre_trained_weights = './vgg16.npy'
    data_dir = './data/cifar-10-batches-bin/'
    train_log_dir = './logs/train/'
    val_log_dir = './logs/val'

    with tf.name_scope('input'):
        tra_image_batch, tra_label_batch = input_cifar.read_cifar10(data_dir=data_dir,
                                                              is_train = True,
                                                              batch_size = BATCH_SIZE,
                                                              shuffle = True)
        val_image_batch, val_label_batch = input_cifar.read_cifar10(data_dir=data_dir,
                                                                    is_train = False,
                                                                    batch_size = BATCH_SIZE,
                                                                    shuffle = False)

    logits = net_define.VGG16(tra_image_batch, N_CLASSES)
    loss = functions.loss(logits, tra_label_batch)
    accuracy = functions.accuracy(logits, tra_label_batch)
    my_global_step = tf.Variable(0,name='global_step', trainable=False)
    train_op = functions.optimize(loss, learning_rate, my_global_step)

    x = tf.placeholder(tf.float32,shape=[BATCH_SIZE, IMG_W,IMG_H,3])
    y_ = tf.placeholder(tf.float16, shape=[BATCH_SIZE, N_CLASSES])

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)


    #load pretrained model
    if tf.gfile.Exists(pre_trained_weights):
        functions.load_conv(pre_trained_weights, sess, ['fc6','fc7','fc8'])
        print ('load VGG model!')


    ckpt = tf.train.get_checkpoint_state(train_log_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
        print ('Loading successfully, globa; step is % d'%my_global_step.eval(sess))
    else:
        print ("No checkpoint file!")


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    tra_summary_writer = tf.summary.FileWriter(train_log_dir,sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir)

    try:
        for step in np.arange(MAX_STEP):
            glb_step = my_global_step.eval(sess)
            if coord.should_stop():
                break

            tra_images, tra_labels = sess.run([tra_image_batch, tra_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x: tra_images, y_ : tra_labels})

            if step % 10 ==0 or (step + 1) ==MAX_STEP:
                print ('Globale step: %d, Step: %d, loss: %.4f, accuracy: %.4f'%(glb_step, step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op)
                tra_summary_writer.add_summary(summary_str, step)

            if step % 100 ==0 or (step + 1) ==MAX_STEP:
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={x:val_images,y_:val_labels})
                print ("   Global step: %d, Step %d, val loss = %.2f, val accuracy = %.2f " %(glb_step, step, val_loss, val_acc))

                summary_str = sess.run(summary_op)
                val_summary_writer.add_summary(summary_str,step)


            if step % 10 ==0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir,'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=glb_step)


    except tf.errors.OutOfRangeError:
        print ("Done training --out of LIMIT epochs")
        coord.request_stop()

    finally:
        coord.request_stop()
        coord.join(threads)
    sess.close()


if __name__ == '__main__':
    train()


