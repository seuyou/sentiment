import tensorflow as tf 
import numpy as np 


class Extract_senti(object):

    def __init__(self, max_len, output_dim, embedding_size=300):

        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, max_len, embedding_size], name="input_x")
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, max_len], name="input_y")
        self.shape = tf.shape(self.input_x)[0]

        self.W = tf.get_variable("transformation_matrix", [embedding_size, output_dim], dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
        


        with tf.name_scope("linear_mapping"):

                input_x_reshape = tf.reshape(self.input_x, (-1, embedding_size))
                mapping = tf.reshape(tf.matmul(input_x_reshape, self.W), (-1, max_len, output_dim))
                mapping_expanded = tf.expand_dims(mapping, -1)

        with tf.name_scope("maxpool"):

                pool = tf.nn.max_pool(mapping_expanded,
                                      ksize=[1, 1, output_dim, 1],
                                      strides=[1, 1, 1, 1],
                                      padding="VALID",
                                      name="pool"
                        )
                pool_reshape = tf.reshape(pool, (-1, max_len))
                

        with tf.name_scope("loss"):
                
                self.record = tf.sigmoid(pool_reshape)
                self.loss = tf.reduce_sum(tf.pow(self.input_y, self.record))/self.shape
                #self.loss = tf.losses.mean_squared_error(self.input_y, self.record)

                





