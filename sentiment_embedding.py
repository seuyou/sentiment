import tensorflow as tf 

def SentimentCNN(object):

    def __init__(self, max_len, filter_size, num_filters, embedding_size=300, loss_weight, l2_reg_lambda=0.0):

        self.input_x = tf.placeholder(tf.float32, [None, max_len, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)

        with tf.device('/gpu:0'), tf.name_scope("Convolution"):

            W = tf.get_variable("Kernel", 
                                [filter_size, filter_size, 1, num_filters], 
                                dtype=tf.float32, 
                                initializer=tf.random_normal_initializer)
            b = tf.get_variable("bias", 
                                [num_filters], 
                                dtype=tf.float32,
                                initializer=tf.constant_initializer([0.1]))
            cnn = tf.nn.conv2d(self.input_x, 
                               W,
                               strides=[1, 1, 1, 1],
                               padding="VALID")

            h = tf.nn.relu(tf.nn.bias_add(cnn, b), name="relu")

            pooled = tf.nn.max_pool(h,
                                    ksize=[1, 1, embedding_size-filter_size+1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding="VALID")
            
            num_filters_total = (max_len-filter_size+1)*num_filters
            self.h_pool_flat = tf.reshape(pooled, shape=[-1, num_filters_total])
        
        with tf.name_scope("drop_out"):

            self.h_dropout = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_drop)

        with tf.name_scope("Feed_forward"):

            W_f = tf.get_variable("mapping",
                                  [num_filters_total, embedding_size],
                                  dtype=tf.float32,
                                  initializer=tf.random_normal_initializer)
            b_f = tf.get_variable("bias_f",
                                   [embedding_size],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer([0.1]))
            self.score = tf.nn.xw_plus_b(self.h_dropout, W_f, b_f, name="scores")

            l2_loss += tf.nn.l2_loss(W_f)
            l2_loss += tf.nn.l2_loss(b_f)



        with tf.name_scope("loss"):
            
            self.loss = tf.losses.mean_squared_error(self.input_y, self.score, weights=loss_weight)
            




