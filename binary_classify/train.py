import os
import numpy as np
import tensorflow as tf
import input
import model

N_CLASSES = 2
IMG_W = 227
IMG_H = 227
RATIO = 0.1
BATCH_SIZE = 128
CAPACITY = 2000
MAX_STEP = 20000
learning_rate = 0.0001


def train():
    train_pos_dir = '../data/train_pos/'
    train_neg_dir = '../data/train_neg/'
    logs_train_dir = './logs/train/'
    logs_val_dir = './logs/val/'

    train, train_label, val, val_label = input.get_files(train_pos_dir, train_neg_dir,  RATIO)
    train_batch, train_label_batch = input.get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    val_batch, val_label_batch = input.get_batch(val, val_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    dp = tf.placeholder(tf.float32,name = 'Dropout_rate')
    logits = model.model(x, BATCH_SIZE, N_CLASSES, dp)
    loss = model.losses(logits, y_)
    train_op,gbstp = model.trainning(loss, learning_rate)

    acc = model.evaluation(logits, y_)




    with tf.Session() as sess:
        saver = tf.train.Saver()

        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found,initialize paras.')
            sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        val_writer = tf.summary.FileWriter(logs_val_dir)

        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break

                tra_images, tra_labels = sess.run([train_batch, train_label_batch])
                sess.run(train_op, feed_dict={x: tra_images, y_: tra_labels, dp:0.5})
                if step % 10 == 0:
                    summary_str, tra_loss, tra_acc = sess.run([summary_op, loss, acc],
                                                    feed_dict={x: tra_images, y_: tra_labels, dp: 1.})
                    print('Step %d, train loss = %.5f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                    train_writer.add_summary(summary_str, step)

                if step % 100 == 0 or (step + 1) == MAX_STEP:
                    val_images, val_labels = sess.run([val_batch, val_label_batch])
                    summary_str,val_loss, val_acc = sess.run([summary_op, loss, acc],
                                                 feed_dict={x: val_images, y_: val_labels, dp: 1.})
                    print('**  Step %d, val loss = %.5f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc * 100.0))
                    val_writer.add_summary(summary_str, step)

                if step % 500 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=gbstp.eval(sess))

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
            coord.request_stop()
        finally:
            coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    train()
