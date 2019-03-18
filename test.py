from first_layer_datahelper import training_data_pay
import tensorflow as tf
import numpy as np
from data_helpers import batch_iter
class Extract_senti(object):

    def __init__(self, max_len, output_dim, embedding_size=300):

        self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, max_len, embedding_size], name="input_x")
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, max_len], name="input_y")
        

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
                self.pool_reshape = tf.reshape(pool, (-1, max_len))


        with tf.name_scope("loss"):

                self.record = tf.sigmoid(self.pool_reshape)
                self.loss = tf.losses.mean_squared_error(self.input_y, self.record)



def main(argv=None):
    max_len = 10
    
    vector_file = "/seu_share/home/txli/YuanLiu/dataset/wiki-news.vec"
    file_csv = "/seu_share/home/txli/YuanLiu/dataset/words_scores.csv"
    train_x, train_y, dev_x, dev_y = training_data_pay(file_csv, vector_file, max_len, False)
    print("shape of train_x:{}".format(np.shape(train_x)))
    batches = batch_iter(train_x, train_y, 30, 200)
    cnn = Extract_senti(max_len, 300)
    for batch in batches:
        train_x_test, train_y_test = batch
        for i in range(10):
            train_x_test_c = np.reshape(train_x_test[i], (1, max_len, 300))
            train_y_test_c = np.reshape(train_y_test[i], (1, max_len))           
            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init)
            pool, loss = sess.run([cnn.pool_reshape, cnn.loss], feed_dict={cnn.input_x:train_x_test_c, cnn.input_y:train_y_test_c} )
            print(pool)
            print(train_y_test)
            print(loss)
        break

if __name__ == "__main__":
    tf.app.run()



