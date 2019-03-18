import tensorflow as tf 
import pandas as pd 
import numpy as np 
import os
import time
from data_helpers import enumerate_append, batch_iter
from gensim.models import KeyedVectors
from first_layer_datahelper import process_csv, text_to_vector, load_word2vec, training_data_pay
from sentiment_extract import Extract_senti
import datetime



tf.flags.DEFINE_string("csv_file", "./WordSentiScore.csv", "where the csv file is located")
tf.flags.DEFINE_float("dev_portion", "0.1", "The propotion of training data used to train")


FLAGS = tf.flags.FLAGS

max_len = 80


def train(x_train, y_train, x_dev, y_dev):

    
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True
    )
    sess = tf.Session(config=session_config)
    
    senti = Extract_senti(
        max_len=max_len,
        output_dim=300,
        embedding_size=300
    )
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(senti.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            grad_summaries.append(grad_hist_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}".format(out_dir))

    loss_summary = tf.summary.scalar("loss", senti.loss)
    

    train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)


    dev_summary_op = tf.summary.merge([loss_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    check_point_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(check_point_dir, "model")

    if not os.path.exists(check_point_dir):
        os.makedirs(check_point_dir)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    sess.run(tf.global_variables_initializer())

    def train_step(x_batch, y_batch):

        feed_dict = {
            senti.input_x: x_batch,
            senti.input_y: y_batch
            }
        _, step, summaries, loss = sess.run([train_op, global_step, train_summary_op, senti.loss], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step{} , loss {:g}".format(time_str, step, loss))
        train_summary_writer.add_summary(summaries, step)

    def dev_step(x_batch, y_batch):

        feed_dict = {
            senti.input_x:x_batch,
            senti.input_y:y_batch
        }
        step, summaries, loss = sess.run([global_step, dev_summary_op, senti.loss], feed_dict=feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step{}, loss {:g}".format(time_str, step, loss))
        dev_summary_writer.add_summary(summaries, step)


    batches = batch_iter(x_train, y_train, 30, 200)

    for batch in batches:

        x_batch, y_batch = batch   
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)

        if current_step % 10 == 0:
            print("\nEvaluation:")
            dev_step(x_dev, y_dev)

        if current_step % 100 == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Save model to checkpoint {}\n".format(path))



def main(argv=None):
    
    vector_file = "/seu_share/home/txli/YuanLiu/dataset/wiki-news.vec"
    file_csv = "/seu_share/home/txli/YuanLiu/dataset/words_scores.csv"
    train_x, train_y, dev_x, dev_y = training_data_pay(file_csv, vector_file, max_len, False)
    print("train_x:{}".format(np.shape(train_x)))
    train(train_x, train_y, dev_x, dev_y)



if __name__ == "__main__":
    tf.app.run()


                






            





            





