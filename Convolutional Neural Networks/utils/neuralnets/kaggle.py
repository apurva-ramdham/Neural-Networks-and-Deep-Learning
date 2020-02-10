#!/usr/bin/env/ python
# ECBM E4040 Fall 2018 Assignment 2
# TensorFlow custom CNN model

import tensorflow as tf
import numpy as np
import time
from utils.neuralnets.cnn.layers import conv_layer,max_pooling_layer, norm_layer, fc_layer
from utils.image_generator import ImageGenerator


def kaggle_Net(input_x, input_y, is_training,
          img_len=28, channel_num=1, output_size=10,
          conv_featmap=[6, 16], fc_units=[84],
          conv_kernel_size=[5, 5], pooling_size=[2, 2],
          l2_norm=0.01, seed=235):
    #raise NotImplementedError
    assert len(conv_featmap) == len(conv_kernel_size) and len(conv_featmap) == len(pooling_size)

    # conv layer
    conv_layer_0 = conv_layer(input_x=input_x,
                              in_channel=channel_num,
                              out_channel=conv_featmap[0],
                              kernel_shape=conv_kernel_size[0],
                              rand_seed=seed)

    pooling_layer_0 = max_pooling_layer(input_x=conv_layer_0.output(),
                                        k_size=pooling_size[0],
                                        padding="VALID")

    # flatten
    pool_shape = pooling_layer_0.output().get_shape()
    img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(pooling_layer_0.output(), shape=[-1, img_vector_length])

    # fc layer
    fc_layer_0 = fc_layer(input_x=flatten,
                          in_size=img_vector_length,
                          out_size=fc_units[0],
                          rand_seed=seed,
                          activation_function=tf.nn.relu,
                          index=0)

    fc_layer_1 = fc_layer(input_x=fc_layer_0.output(),
                          in_size=fc_units[0],
                          out_size=output_size,
                          rand_seed=seed,
                          activation_function=None,
                          index=1)
    # output
    out = fc_layer_1.output()
    # saving the parameters for l2_norm loss
    conv_w = [conv_layer_0.weight]
    fc_w = [fc_layer_0.weight, fc_layer_1.weight]

    # loss
    with tf.name_scope("loss"):
        l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])
        l2_loss += tf.reduce_sum([tf.reduce_sum(tf.norm(w, axis=[-2, -1])) for w in conv_w])

        label = tf.one_hot(input_y, 10)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=out),
            name='cross_entropy')
        loss = tf.add(cross_entropy_loss, l2_norm * l2_loss, name='loss')

        tf.summary.scalar('LeNet_loss', loss)

    return out, loss

def cross_entropy(output, input_y):
    with tf.name_scope('cross_entropy'):
        label = tf.one_hot(input_y, 10)
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=output))

    return ce

def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('kaggle_net_error_num', error_num)
    return error_num

def train_step(loss, learning_rate=1e-3):
    with tf.name_scope('train_step'):
        step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return step
    

def kaggle_training(X_train, y_train, X_val, y_val,
                conv_featmap=(6, 16),
                fc_units=(120, 84),
                conv_kernel_size=(5, 5),
                pooling_size=(2, 2),
                l2_norm=0.01,
                seed=235,
                learning_rate=1e-3,
                epoch=20,
                batch_size=295,
                verbose=False,
                pre_trained_model=None):

    print("Building kaggle Net. Parameters: ")
    print("conv_featmap={}".format(conv_featmap))
    print("fc_units={}".format(fc_units))
    print("conv_kernel_size={}".format(conv_kernel_size))
    print("pooling_size={}".format(pooling_size))
    print("l2_norm={}".format(l2_norm))
    print("seed={}".format(seed))
    print("learning_rate={}".format(learning_rate))

    img_len = X_train.shape[1]
    channel_num = X_train.shape[-1]
    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, img_len, img_len, channel_num], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)
        is_training = tf.placeholder(tf.bool, name='is_training')

    output, loss = kaggle_Net(xs, ys, is_training,
                         img_len=img_len,
                         channel_num=channel_num,
                         output_size=10,
                         conv_featmap=conv_featmap,
                         fc_units=fc_units,
                         conv_kernel_size=conv_kernel_size,
                         pooling_size=pooling_size,
                         l2_norm=l2_norm,
                         seed=seed)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    step = train_step(loss, learning_rate)
    eve = evaluate(output, ys)

    iter_total = 0
    best_acc = 0
    cur_model_name = 'KaggleModel'
    #cur_model_name = 'best_LeNet_model'

    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                raise ValueError("Load model Failed!")

        for epc in range(epoch):
            print("epoch {} ".format(epc + 1))

            for itr in range(iters):
                iter_total += 1

                training_batch_x = X_train[itr * batch_size: (1 + itr) * batch_size]
                training_batch_y = y_train[itr * batch_size: (1 + itr) * batch_size]

                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x,
                                                                ys: training_batch_y,
                                                                is_training: True})

                if iter_total % 100 == 0:
                    # do validation
                    valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val,
                                                                                ys: y_val,
                                                                                is_training: False})
                    valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
                    if verbose:
                        print('{}/{} loss: {} validation accuracy : {}%'.format(
                            batch_size * (itr + 1),
                            X_train.shape[0],
                            cur_loss,
                            valid_acc))

                    # save the merge result summary
                    writer.add_summary(merge_result, iter_total)
                    print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                    # when achieve the best validation accuracy, we store the model paramters
                    if valid_acc > best_acc:
                        
                        best_acc = valid_acc
                        saver.save(sess, 'model/{}'.format(cur_model_name))

    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))

def predict(output):
  with tf.name_scope('predict'):
      pred = tf.argmax(output, axis=1)
  return pred



def kaggle_testing(X_test,pre_trained_model,batch_size):
# define the variables and parameter needed during testing
  N = X_test.shape[0]
  with tf.name_scope('inputs'):
      xs = tf.placeholder(shape=[None,128, 128, 3], dtype=tf.float32)
      ys = tf.placeholder(shape=[None, ], dtype=tf.int64)
      is_training = tf.placeholder(tf.bool, name='is_training')

  output, _ = kaggle_Net(xs, ys, is_training,
                       img_len=128,
                       channel_num=3,
                       output_size=5,
                       conv_featmap=[64],
                       fc_units=[100],
                       conv_kernel_size=[32],
                       pooling_size=[16],
                       l2_norm=0.01,
                       seed=235)

  pred = predict(output)
  y_test = []
  saver = tf.train.Saver()
  iters = int(X_test.shape[0] / batch_size)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'model/{}'.format(pre_trained_model))
    
    for itr in range(iters):
      test_batch_x = X_test[itr * batch_size: (1 + itr) * batch_size]
      temp = sess.run([pred], feed_dict={xs: test_batch_x,
                                         ys: np.ones(test_batch_x.shape[0],),
                                         is_training: False})
      for j in range(len(temp[0])):
        y_test.append(temp[0][j])

  y_test = np.array(y_test)
  return y_test