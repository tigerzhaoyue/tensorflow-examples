import tensorflow as tf


def model(images, batch_size, n_classes,dp):
    '''

    :param images:image batch 
    :param batch_size: batch size
    :param n_classes: 2
    :return: logits [batch_size, n_classes]
    '''

    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights',
                                  shape=[11, 11, 3, 96],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        bbiases = tf.get_variable('biases',
                                  shape=[96],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))

        net = tf.nn.conv2d(images, weights, strides=[1, 4, 4, 1], padding='SAME')
        net = tf.nn.bias_add(net, bbiases)
        net = tf.nn.relu(net, name=scope.name)

    with tf.variable_scope('pool1_lrn') as scope:
        net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')
        net = tf.nn.lrn(net, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[5, 5, 96, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, biases)
        net = tf.nn.relu(net, name=scope.name)

    with tf.variable_scope('pooling2_lrn') as scope:
        net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pooling2')
        net = tf.nn.lrn(net, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                        beta=0.75, name='norm2')

    with tf.variable_scope('conv3') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 256, 384],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[384],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, biases)
        net = tf.nn.relu(net, name=scope.name)



    with tf.variable_scope('conv4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 384, 384],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[384],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, biases)
        net = tf.nn.relu(net, name=scope.name)


    with tf.variable_scope('conv5') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 384, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, biases)
        net = tf.nn.relu(net, name=scope.name)


    with tf.variable_scope('fc6') as scope:
        net = tf.reshape(net, shape=[batch_size, -1])
        dim = net.get_shape()[1].value
        print '\n\n\n',dim,'\n\n\n'
        weights = tf.get_variable('weights',
                                  shape=[dim, 4096],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[4096],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        net = tf.matmul(net, weights) + biases
        net = tf.nn.relu(net, name=scope.name)
        net = tf.nn.dropout(net,dp)
        l2_loss = tf.nn.l2_loss(weights)

    with tf.variable_scope('fc7') as scope:
        weights = tf.get_variable('weights',
                                  shape=[4096, 1024],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[1024],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        net = tf.matmul(net, weights) + biases
        net = tf.nn.relu(net, name=scope.name)
        net = tf.nn.dropout(net, dp)
        l2_loss += tf.nn.l2_loss(weights)


    # softmax classifier
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[1024, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))

        net = tf.add(tf.matmul(net, weights), biases, name='softmax_linear')
        l2_loss += tf.nn.l2_loss(weights)

    return net,l2_loss


def losses(logits, labels, l2_loss, lamda):
    '''

    :param logits: 
    :param labels: 
    :return: scalar loss tensor
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels,
                                                                       name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='x_loss')
        tf.summary.scalar(scope.name + '/xentro_loss', loss)
        tf.summary.scalar(scope.name+ '/l2_loss',l2_loss)
        total_loss = loss + lamda * l2_loss
        tf.summary.scalar(scope.name + '/total_loss', total_loss)
    return total_loss


def trainning(loss, learning_rate):
    '''

    :param loss: 
    :param learning_rate: 
    :return: 
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op, global_step


def evaluation(logits, labels):
    '''

    :param logits: 
    :param labels: 
    :return: 
    '''
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
