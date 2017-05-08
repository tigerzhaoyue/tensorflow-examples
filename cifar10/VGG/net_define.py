import tensorflow as tf
import functions
# %%
def VGG16(x, n_classes, trainable=True):
    x = functions.conv('conv1_1', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], trainable=trainable)
    x = functions.conv('conv1_2', x, 64, kernel_size=[3, 3], stride=[1, 1, 1, 1], trainable=trainable)
    x = functions.pool('pool1', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = functions.conv('conv2_1', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], trainable=trainable)
    x = functions.conv('conv2_2', x, 128, kernel_size=[3, 3], stride=[1, 1, 1, 1], trainable=trainable)
    x = functions.pool('pool2', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = functions.conv('conv3_1', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], trainable=trainable)
    x = functions.conv('conv3_2', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], trainable=trainable)
    x = functions.conv('conv3_3', x, 256, kernel_size=[3, 3], stride=[1, 1, 1, 1], trainable=trainable)
    x = functions.pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = functions.conv('conv4_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], trainable=trainable)
    x = functions.conv('conv4_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], trainable=trainable)
    x = functions.conv('conv4_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], trainable=trainable)
    x = functions.pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = functions.conv('conv5_1', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], trainable=trainable)
    x = functions.conv('conv5_2', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], trainable=trainable)
    x = functions.conv('conv5_3', x, 512, kernel_size=[3, 3], stride=[1, 1, 1, 1], trainable=trainable)
    x = functions.pool('pool3', x, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

    x = functions.FC_layer('fc6', x, out_nodes=4096)
    x = functions.batch_norm(x)
    x = functions.FC_layer('fc7', x, out_nodes=4096)
    x = functions.batch_norm(x)
    x = functions.FC_layer('fc8', x, out_nodes=n_classes)

    return x