from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import model
import input
import shutil
import csv
image_W = 227
image_H = 227
def get_one_image(train):
   '''
   Randomly pick one image from training data
   Return: ndarray
   '''
   n = len(train)
   ind = np.random.randint(0, n)
   img_dir = train[ind]

   print img_dir

   image = Image.open(img_dir)
   plt.imshow(image)
   image = image.resize([image_W, image_H])
   image = np.array(image)
   return image


def get_by_name(name):
    image = Image.open(name)
    image = image.resize([image_W, image_H])
    image = np.array(image)
    return image


def evaluate_one_image(name):
   '''Test one image against the saved models and parameters
   '''

   # you need to change the directories to yours.

   #image_array = get_one_image(test)

   with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 2
       name = tf.cast(name, tf.string)
       image_raw = tf.read_file(name)
       image = tf.image.decode_jpeg(image_raw, channels=3)
       #image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
       image = tf.image.resize_images(image, [image_W, image_H], method=tf.image.ResizeMethod.AREA)
       image = tf.cast(image, tf.float32)
       #image = tf.image.per_image_standardization(image)
       image = tf.reshape(image, [1, image_W, image_H, 3])
       dp = tf.placeholder(tf.float32)
       logit,total_loss = model.model(image, BATCH_SIZE, N_CLASSES, dp)

       logit = tf.nn.softmax(logit)


       # you need to change the directories to yours.
       logs_train_dir = './logs/train/'

       saver = tf.train.Saver()

       with tf.Session() as sess:

           print("Reading checkpoints...")
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)
           if ckpt and ckpt.model_checkpoint_path:
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               saver.restore(sess, ckpt.model_checkpoint_path)
               print('Loading success, global_step is %s' % global_step)
           else:
               print('No checkpoint file found')

           prediction = sess.run(logit, feed_dict={dp:1.})
           max_index = np.argmax(prediction)
           if max_index==0:
               print('\t\t\tNEG\t %.6f\n' %prediction[:, 0])
               return 0,prediction[:, 0]
           else:
               print('\t\t\tPOS\t %.6f\n' %prediction[:, 1])
               return 1, prediction[:, 1]


if __name__== '__main__':
    test_pos_dir = '../data/test_pos/'
    test_neg_dir = '../data/test_neg/'
    # train, train_label = input.get_files(train_dir,0.2)
    test, test_label, val, val_label = input.get_files(test_pos_dir, test_neg_dir, 0)
    n=0
    y=0
    x=0
    csvfile = open('./result/result.csv','a')
    writer = csv.writer(csvfile)
    for name in test:
        row=[]
        pred, prob = evaluate_one_image(name)
        if pred ==1:
            x = x+1
            shutil.copy(name, './result/pos/')
            row.append('POS')
        else:
            y = y+1
            shutil.copy(name, './result/neg/')
            row.append('NEG')
        row.append(name.split('/')[-1])
        row.append(str(prob[0]))
        writer.writerow(row)
        #plt.imshow(image)
        #plt.show()
        n=n+1
    print x
    print y
    print n




