import argparse

import io

import types

import logging

import numpy as np
import seaborn as sn

import time

%matplotlib inline
import matplotlib.pyplot as plt

sn.set()

import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer

tf.reset_default_graph()

# import resource

import signal

import sys


from YahooExp_util_functions import getClusters, getIDAssignment, parseLine, save_to_file, articleAccess

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG)

x_result = []
y_result = []

class MultiVAE():
    def __init__(self, p_dims, q_dims=None, lam=0.01, lr=5e-5, random_seed=98765):
        self.p_dims = p_dims

        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims
        self.dims = self.q_dims + self.p_dims[1:]

        self.lam = lam
        self.lr = lr
        self.random_seed = random_seed

        self.construct_placeholders()
        self.construct_weights()

    def construct_placeholders(self):

        # placeholders with default values when scoring
        self.is_training_ph = tf.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf.placeholder_with_default(1., shape=None)
        self.input_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.dims[0]])
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.article_features = tf.placeholder(dtype=tf.float32, shape=[None, self.dims[0]])
        self.epsilon = tf.placeholder(dtype=tf.float32, shape=[None, self.dims[0]])

    def build_graph(self):

        mu_q, std_q, KL = self.q_graph()

        sampled_z = mu_q + self.is_training_ph * \
                    self.epsilon * std_q
        
        # p-network
        logits = self.p_graph(sampled_z)

        '''
        neg_ll = -tf.reduce_sum(self.input_ph * tf.log(logits) +
        (1 - self.input_ph) * tf.log(1 - logits), 1)
        '''
        neg_ll = tf.reduce_sum(tf.square(tf.nn.l2_normalize(self.input_ph, 1) - logits), 1)

        # apply regularization to weights
        reg = l2_regularizer(self.lam)

        neg_reward = tf.reduce_sum(sampled_z * self.article_features, 1)

        reg_var = apply_regularization(reg, self.weights_q + self.weights_p)
        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        neg_ELBO = neg_ll + self.anneal_ph * KL + 2 * reg_var - tf.multiply(self.reward, neg_reward)

        train_op = tf.train.AdamOptimizer(self.lr).minimize(neg_ELBO)

        # add summary statistics
        tf.summary.scalar('negative_multi_ll', neg_ll)
        tf.summary.scalar('KL', KL)
        tf.summary.scalar('neg_ELBO_train', neg_ELBO)
        merged = tf.summary.merge_all()

        return logits, neg_ELBO, train_op, merged, sampled_z

    def q_graph(self):

        mu_q, std_q, KL = None, None, None

        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)

        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf.matmul(h, w) + b

            if i != len(self.weights_q) - 1:
                h = tf.nn.tanh(h)
            else:
                h = tf.sigmoid(h)
                mu_q = h[:, :self.q_dims[-1]]
                logvar_q = h[:, self.q_dims[-1]:]

                std_q = tf.exp(0.5 * logvar_q)
                KL = tf.reduce_mean(tf.reduce_sum(
                    0.5 * (-logvar_q + tf.exp(logvar_q) + mu_q ** 2 - 1), axis=1))
        return mu_q, std_q, KL

    def p_graph(self, z):
        h = z

        for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
            h = tf.matmul(h, w) + b

            if i != len(self.weights_p) - 1:
                h = tf.nn.tanh(h)
            else:
                h = tf.sigmoid(h)
        return h
      
      

    def forward_pass(self):
        # q-network
        mu_q, std_q, KL = self.q_graph()
        epsilon = tf.random_normal(tf.shape(std_q))

        sampled_z = mu_q + self.is_training_ph * \
                    epsilon * std_q

        # p-network
        logits = self.p_graph(sampled_z)

        return tf.train.Saver(), logits, KL, sampled_z, epsilon

    def construct_weights(self):

        self.weights_q, self.biases_q = [], []

        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                # we need two sets of parameters for mean and variance,
                # respectively
                d_out *= 2
            weight_key = "weight_q_{}to{}".format(i, i + 1)
            bias_key = "bias_q_{}".format(i + 1)

            self.weights_q.append(tf.Variable(tf.random_normal([d_in, d_out])))

            self.biases_q.append(tf.Variable(tf.random_normal([d_out])))

            # add summary stats
            tf.summary.histogram(weight_key, self.weights_q[-1])
            tf.summary.histogram(bias_key, self.biases_q[-1])

        self.weights_p, self.biases_p = [], []

        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            weight_key = "weight_p_{}to{}".format(i, i + 1)
            bias_key = "bias_p_{}".format(i + 1)
            self.weights_p.append(tf.Variable(tf.random_normal([d_in, d_out])))

            self.biases_p.append(tf.Variable(tf.random_normal([d_out])))

            # add summary stats
            tf.summary.histogram(weight_key, self.weights_p[-1])
            tf.summary.histogram(bias_key, self.biases_p[-1])


d = 6

alpha = 5

r1 = 7

r2 = 0

n_epochs = 4

batch_size = 40


def update(reward_batch, train_op_var, vae, X, sess, article_batch, epsilon_batch):

    i = 0
    for reward in reward_batch:
        if reward == -1:

            return

        elif reward == 1:

            reward_batch[i] = r1

        elif reward == 0:

            reward_batch[i] = r2
        i += 1

    reward_batch = reward_batch.reshape([batch_size, 1])
    update_count = 0.0

    for epoch in range(n_epochs):
        feed_dict = {vae.input_ph: X,
                     vae.keep_prob_ph: 1,
                     vae.anneal_ph: 0.5,
                     vae.is_training_ph: 1,
                     vae.reward: reward_batch,
                     vae.article_features: article_batch,
                     vae.epsilon: epsilon_batch}
        
        sess.run(train_op_var, feed_dict=feed_dict)


def evaluate(input_generator, train_op_var, vae, sampled_z, sess, epsilon):
    score1, score2 = 0.0, 0

    impressions1, impressions2, batch_num = 0.0, 0, 0

    n_lines = 0.0

    article_batch = np.zeros([batch_size, d])
    user_batch = np.zeros([batch_size, d])
    reward_batch = np.zeros(batch_size)
    epsilon_batch = np.zeros([batch_size, d])

    for line in input_generator:

        n_lines += 1

        tim, article_chosen, reward, user_features, pool_articles = parseLine(line)

        user_features = np.array(user_features)

        feed_dict = {vae.input_ph: user_features.reshape(1, d)}
        mu, epsilon_2 = sess.run([sampled_z, epsilon], feed_dict=feed_dict)


        value, max_a = 0, 0

        for article in pool_articles:
            article_id = article[0]

            article_featureVector = np.asarray(article[1:7])

            if article_featureVector.shape[0] == 6:

               value_0 = np.matmul(mu, article_featureVector)

            if value_0 > value:
               max_a = article_id
               value = value_0
               article_features = article_featureVector

        if max_a == article_chosen:

            article_batch[batch_num] = article_features
            reward_batch[batch_num] = reward
            user_batch[batch_num] = user_features
            epsilon_batch[batch_num] = epsilon_2
            batch_num += 1

            if batch_num == batch_size:
                update(reward_batch, train_op_var, vae, user_batch, sess, article_batch, epsilon_batch)
                batch_num = 0

            score1 += reward

            impressions1 += 1

            score2 += reward

            impressions2 += 1

            if impressions1 == 200:
                score1 /= impressions1
                print("CTR for this epoch: %.5f" % score1)
                x_result.append(impressions2 // 200)
                y_result.append(score1)

                score1 = 0
                impressions1 = 0


    if impressions2 < 1:

            logger.info("No impressions were made.")

            return 0.0

    else:

            score2 /= impressions2

            logger.info("CTR achieved by the policy: %.5f" % score2)

            return score2

def run(log_file, train_op_var, vae, sampled_z, sess, epsilon):
    with open(log_file, 'r', buffering=1024 * 1024 * 512) as inf:
        return evaluate(inf, train_op_var, vae, sampled_z, sess, epsilon)


if __name__ == "__main__":

    log_file = "./data/ydata-fp-td-clicks-v1_0.20090501"
    articles_file = "drive/My Drive/Colab Notebooks/Bandit_vae/data/webscope-articles.txt"
    source_file = "policy.py"

    p_dims = [6, 12, 36, 6]

    # saver, logits_var, loss_var, train_op_var, merged_var, sampled_z = vae.build_graph()
    tf.reset_default_graph()
    vae = MultiVAE(p_dims, lam=0.02, random_seed=98765)
    
    logits_var, loss_var, train_op_var, merged_var, _ = vae.build_graph()
    _, logits, KL, sampled_z, epsilon = vae.forward_pass()
    
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.local_variables_initializer())

        starttime = time.time()
        run(log_file, train_op_var, vae, sampled_z, sess, epsilon)
        endtime = time.time()
        print(endtime - starttime)

        plt.figure()  # 创建绘图对象

        plt.plot(x_result, y_result, "b-", linewidth=1)  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）

        plt.xlabel("Iteration(200)")  # X轴标签

        plt.ylabel("CTR")  # Y轴标签

        plt.title("batch_train")  # 图标题

        plt.show()  # 显示图

        plt.savefig("batch_train.jpg")  # 保存

        sess.close()
