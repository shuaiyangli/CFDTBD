import time
import tensorflow as tf
import os
import sys
from load_data import Data
import numpy as np
import math
import multiprocessing
import heapq
import random as rd

# from sklearn.linear_modal import LogisticRegression
def seed_everything(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    rd.seed(seed)

seed_everything(2024)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_name = 'CFDTBD'
data_path = '../Data/'
dataset = 'Taobao'

'''
#######################################################################
Hyper-parameter settings.
'''

n_layers = 5
decay = 0.001
interval = 5
lambda_v, lambda_t, lambda_m = 0.6, 0.3, 1.0
mju_mf_v, mju_mf_t, mju_mf_m = 1.2, 1.0, 1.9
mju_emb_v, mju_emb_t, mju_emb_m = 1.1, 0.4, 1.0
eit_dfd, eit_cfd = 0.08, 1.0

lr = 0.001
batch_size = 2048
embed_size = 64
epoch = 500
data_generator = Data(path=data_path + dataset, batch_size=batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = batch_size
Ks = np.arange(1, 21)

# model test module
def test_one_user(x):
    seed_everything(2024)
    u, rating = x[1], x[0]

    training_items = data_generator.train_items[u]

    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = rd.sample(list(all_items - set(training_items) - set(user_pos_test)), 99) + user_pos_test

    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)

    precision, recall, ndcg, hit_ratio = [], [], [], []

    def hit_at_k(r, k):
        r = np.array(r)[:k]
        if np.sum(r) > 0:
            return 1.
        else:
            return 0.

    def ndcg_at_k(r, k):
        r = np.array(r)[:k]

        if np.sum(r) > 0:
            return math.log(2) / math.log(np.where(r == 1)[0] + 2)
        else:
            return 0.

    for K in Ks:
        ndcg.append(ndcg_at_k(r, K))
        hit_ratio.append(hit_at_k(r, K))

    return {'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'k': (u, K_max_item_score[:20])}


def test(sess, model, users, items, batch_size, cores):
    seed_everything(2024)
    result = {'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'k': []}

    pool = multiprocessing.Pool(cores)

    u_batch_size = batch_size * 2

    n_test_users = len(users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):

        start = u_batch_id * u_batch_size

        end = (u_batch_id + 1) * u_batch_size

        user_batch = users[start: end]

        item_batch = items

        rate_batch = sess.run(model.batch_ratings, {model.users: user_batch,
                                                    model.pos_items: item_batch})

        user_batch_rating_uid = zip(rate_batch, user_batch)

        batch_result = pool.map(test_one_user, user_batch_rating_uid)

        count += len(batch_result)

        for re in batch_result:
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
            result['k'].append(re['k'])

    assert count == n_test_users
    pool.close()
    return result


class Model(object):
    def __init__(self, data_config, img_feat, text_feat, d1, d2):
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.d1 = d1
        self.d2 = d2
        self.n_fold = 10
        self.norm_adj = data_config['norm_adj']
        self.norm_adj_com = data_config['norm_adj_com']
        self.norm_adj_dif = data_config['norm_adj_dif']
        self.norm_adj_m = data_config['norm_adj_m']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = data_config['lr']
        self.emb_dim = data_config['embed_size']
        self.batch_size = data_config['batch_size']
        self.n_layers = data_config['n_layers']
        self.decay = data_config['decay']

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.weights = self._init_weights()

        '''
        ######################################################################################
        multimodal information processing
        '''
        # Whitening Transformation Module
        t = time.time()
        print('Whitening of pre-trained visual modality features')
        self.im_v1_pre = tf.matmul(img_feat, self.weights['w1_v'])
        self.im_v1 = self.zca_whitening(self.im_v1_pre)

        self.im_v2_pre = tf.matmul(img_feat, self.weights['w2_v'])
        self.im_v2 = self.zca_whitening(self.im_v2_pre)
        print('Already whitening', time.time() - t)

        self.im_t1 = tf.matmul(text_feat, self.weights['w1_t'])
        self.im_t2 = tf.matmul(text_feat, self.weights['w2_t'])


        self.um_v1 = self.weights['user_embedding_v1']
        self.um_t1 = self.weights['user_embedding_t1']

        self.um_v2 = self.weights['user_embedding_v2']
        self.um_t2 = self.weights['user_embedding_t2']

        '''
        ######################################################################################
        generate interactive-dimension embeddings
        '''
        self.ua_embeddings, self.ia_embeddings = self._create_norm_embed()
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        '''
        ######################################################################################
        generate multimodal-dimension embeddings
        '''
        # common feature learning for item visual content
        # self.um_int_v = self.ua_embeddings
        self.ua_embeddings_v1, self.ia_embeddings_v1 = self._create_norm_embed_v1()
        self.u_g_embeddings_v1 = tf.nn.embedding_lookup(self.ua_embeddings_v1, self.users)
        self.pos_i_g_embeddings_v1 = tf.nn.embedding_lookup(self.ia_embeddings_v1, self.pos_items)
        self.neg_i_g_embeddings_v1 = tf.nn.embedding_lookup(self.ia_embeddings_v1, self.neg_items)

        self.u_g_embeddings_v1_pre = tf.nn.embedding_lookup(self.um_v1, self.users)
        self.pos_i_g_embeddings_v1_pre = tf.nn.embedding_lookup(self.im_v1, self.pos_items)
        self.neg_i_g_embeddings_v1_pre = tf.nn.embedding_lookup(self.im_v1, self.neg_items)

        # common feature learning for item textual content
        # self.um_int_t = self.ua_embeddings
        self.ua_embeddings_t1, self.ia_embeddings_t1 = self._create_norm_embed_t1()
        self.u_g_embeddings_t1 = tf.nn.embedding_lookup(self.ua_embeddings_t1, self.users)
        self.pos_i_g_embeddings_t1 = tf.nn.embedding_lookup(self.ia_embeddings_t1, self.pos_items)
        self.neg_i_g_embeddings_t1 = tf.nn.embedding_lookup(self.ia_embeddings_t1, self.neg_items)

        self.u_g_embeddings_t1_pre = tf.nn.embedding_lookup(self.um_t1, self.users)
        self.pos_i_g_embeddings_t1_pre = tf.nn.embedding_lookup(self.im_t1, self.pos_items)
        self.neg_i_g_embeddings_t1_pre = tf.nn.embedding_lookup(self.im_t1, self.neg_items)

        # common feature distillation
        self.cfd_loss = tf.reduce_mean(self.cosine_similarity(self.u_g_embeddings_v1, self.u_g_embeddings_t1)) + \
                        tf.reduce_mean(self.cosine_similarity(self.pos_i_g_embeddings_v1, self.pos_i_g_embeddings_t1))

        # differential feature learning for item visual content
        self.ua_embeddings_v2, self.ia_embeddings_v2 = self._create_norm_embed_v2()
        self.u_g_embeddings_v2 = tf.nn.embedding_lookup(self.ua_embeddings_v2, self.users)
        self.pos_i_g_embeddings_v2 = tf.nn.embedding_lookup(self.ia_embeddings_v2, self.pos_items)
        self.neg_i_g_embeddings_v2 = tf.nn.embedding_lookup(self.ia_embeddings_v2, self.neg_items)

        self.u_g_embeddings_v2_pre = tf.nn.embedding_lookup(self.um_v2, self.users)
        self.pos_i_g_embeddings_v2_pre = tf.nn.embedding_lookup(self.im_v2, self.pos_items)
        self.neg_i_g_embeddings_v2_pre = tf.nn.embedding_lookup(self.im_v2, self.neg_items)

        # differential feature learning for item textual content
        self.ua_embeddings_t2, self.ia_embeddings_t2 = self._create_norm_embed_t2()
        self.u_g_embeddings_t2 = tf.nn.embedding_lookup(self.ua_embeddings_t2, self.users)
        self.pos_i_g_embeddings_t2 = tf.nn.embedding_lookup(self.ia_embeddings_t2, self.pos_items)
        self.neg_i_g_embeddings_t2 = tf.nn.embedding_lookup(self.ia_embeddings_t2, self.neg_items)

        self.u_g_embeddings_t2_pre = tf.nn.embedding_lookup(self.um_t2, self.users)
        self.pos_i_g_embeddings_t2_pre = tf.nn.embedding_lookup(self.im_t2, self.pos_items)
        self.neg_i_g_embeddings_t2_pre = tf.nn.embedding_lookup(self.im_t2, self.neg_items)

        # differential feature distillation
        self.dfd_loss = self.cos_square_dis(self.u_g_embeddings_v2, self.u_g_embeddings_t2) + \
                        self.cos_square_dis(self.pos_i_g_embeddings_v2, self.pos_i_g_embeddings_t2)

        # multimodal fuse
        self.um_m = tf.concat([self.ua_embeddings_v1, self.ua_embeddings_t1, self.ua_embeddings_v2, self.ua_embeddings_t2], axis=1)
        self.im_m = tf.concat([self.ia_embeddings_v1, self.ia_embeddings_t1, self.ia_embeddings_v2, self.ia_embeddings_t2], axis=1)

        self.ua_embeddings_m, self.ia_embeddings_m = self._create_norm_embed_m()
        self.u_g_embeddings_m = tf.nn.embedding_lookup(self.ua_embeddings_m, self.users)
        self.pos_i_g_embeddings_m = tf.nn.embedding_lookup(self.ia_embeddings_m, self.pos_items)
        self.neg_i_g_embeddings_m = tf.nn.embedding_lookup(self.ia_embeddings_m, self.neg_items)

        self.u_g_embeddings_m_pre = tf.nn.embedding_lookup(self.um_m, self.users)
        self.pos_i_g_embeddings_m_pre = tf.nn.embedding_lookup(self.im_m, self.pos_items)
        self.neg_i_g_embeddings_m_pre = tf.nn.embedding_lookup(self.im_m, self.neg_items)

        # visual feature
        self.ua_embeddings_v = tf.concat([self.ua_embeddings_v1, self.ua_embeddings_v2], axis=1)
        self.ia_embeddings_v = tf.concat([self.ia_embeddings_v1, self.ia_embeddings_v2], axis=1)

        self.u_g_embeddings_v = tf.nn.embedding_lookup(self.ua_embeddings_v, self.users)
        self.pos_i_g_embeddings_v = tf.nn.embedding_lookup(self.ia_embeddings_v, self.pos_items)
        self.neg_i_g_embeddings_v = tf.nn.embedding_lookup(self.ia_embeddings_v, self.neg_items)

        # textual feature
        self.ua_embeddings_t = tf.concat([self.ua_embeddings_t1, self.ua_embeddings_t2], axis=1)
        self.ia_embeddings_t = tf.concat([self.ia_embeddings_t1, self.ia_embeddings_t2], axis=1)

        self.u_g_embeddings_t = tf.nn.embedding_lookup(self.ua_embeddings_t, self.users)
        self.pos_i_g_embeddings_t = tf.nn.embedding_lookup(self.ia_embeddings_t, self.pos_items)
        self.neg_i_g_embeddings_t = tf.nn.embedding_lookup(self.ia_embeddings_t, self.neg_items)

        '''
        ######################################################################################
        prediction scores
        '''
        self.batch_ratings_base = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)
        self.batch_ratings_v = tf.matmul(self.u_g_embeddings_v, self.pos_i_g_embeddings_v, transpose_a=False, transpose_b=True)
        self.batch_ratings_t = tf.matmul(self.u_g_embeddings_t, self.pos_i_g_embeddings_t, transpose_a=False, transpose_b=True)
        self.batch_ratings_m = tf.matmul(self.u_g_embeddings_m, self.pos_i_g_embeddings_m, transpose_a=False, transpose_b=True)

        self.batch_ratings = self.batch_ratings_base + lambda_v*self.batch_ratings_v + lambda_t*self.batch_ratings_t + \
                             lambda_m*self.batch_ratings_m

        '''
        ######################################################################################
        loss function
        '''
        self.mf_loss_base, self.emb_loss_base = self.create_bpr_loss_base()
        self.mf_loss_v, self.emb_loss_v = self.create_bpr_loss_v()
        self.mf_loss_t, self.emb_loss_t = self.create_bpr_loss_t()
        self.mf_loss_m, self.emb_loss_m = self.create_bpr_loss_m()

        self.mf_loss = self.mf_loss_base + mju_mf_v*self.mf_loss_v + mju_mf_t*self.mf_loss_t + \
                       mju_mf_m * self.mf_loss_m
        self.emb_loss = self.emb_loss_base + mju_emb_v*self.emb_loss_v + mju_emb_t*self.emb_loss_t + \
                        mju_emb_m*self.emb_loss_m

        self.fd_loss = eit_dfd*self.dfd_loss + eit_cfd*self.cfd_loss

        self.opt_1 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.mf_loss + self.emb_loss + self.fd_loss)

    def _init_weights(self):

        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                    name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                    name='item_embedding')
        all_weights['user_embedding_v1'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                          name='user_embedding_v1')
        all_weights['user_embedding_t1'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                          name='user_embedding_t1')
        all_weights['user_embedding_v2'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                        name='user_embedding_v2')
        all_weights['user_embedding_t2'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                        name='user_embedding_t2')

        all_weights['w1_v'] = tf.Variable(initializer([self.d1, self.emb_dim]), name='w1_v')
        all_weights['w1_t'] = tf.Variable(initializer([self.d2, self.emb_dim]), name='w1_t')

        all_weights['w2_v'] = tf.Variable(initializer([self.d1, self.emb_dim]), name='w2_v')
        all_weights['w2_t'] = tf.Variable(initializer([self.d2, self.emb_dim]), name='w2_t')

        # self.global_T = tf.Variable(initial_value=tf.ones((1,), dtype=tf.float32), trainable=True)
        return all_weights

    def zca_whitening(self, X, batch_size=data_generator.n_items):

        num_batches = int(np.ceil(X.shape[0].value / batch_size))
        whitened_batches = []
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, X.shape[0])
            X_batch = X[start:end]

            cov = tf.matmul(X_batch, X_batch, transpose_a=True) / tf.cast(tf.shape(X_batch)[0], tf.float32)

            s, u, _ = tf.linalg.svd(cov)

            epsilon = 1e-5
            zca_matrix = tf.matmul(tf.matmul(u, tf.linalg.diag(1.0 / tf.sqrt(s + epsilon))), u, transpose_b=True)

            whitened_batch = tf.matmul(X_batch, zca_matrix)
            whitened_batches.append(whitened_batch)

        whitened = tf.concat(whitened_batches, axis=0)
        return whitened

    def _split_A_hat(self, X):

        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _create_norm_embed(self):

        A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat(
            [self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        for k in range(0, self.n_layers):

            temp_embed = []

            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)

            ego_embeddings = side_embeddings

        u_g_embeddings, i_g_embeddings = tf.split(ego_embeddings, [self.n_users, self.n_items], 0)

        return u_g_embeddings, i_g_embeddings

    def _create_norm_embed_v1(self):

        A_fold_hat = self._split_A_hat(self.norm_adj_com)

        ego_embeddings_v = tf.concat([self.um_v1, self.im_v1], axis=0)

        for k in range(0, 1):

            temp_embed = []

            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings_v))

            side_embeddings = tf.concat(temp_embed, 0)

            ego_embeddings_v = side_embeddings

        u_embed, i_embed = tf.split(ego_embeddings_v, [self.n_users, self.n_items], 0)

        return u_embed, i_embed

    def _create_norm_embed_t1(self):

        A_fold_hat = self._split_A_hat(self.norm_adj_com)

        ego_embeddings_t = tf.concat([self.um_t1, self.im_t1], axis=0)

        for k in range(0, 1):

            temp_embed = []

            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings_t))

            side_embeddings = tf.concat(temp_embed, 0)

            ego_embeddings_t = side_embeddings

        u_embed, i_embed = tf.split(ego_embeddings_t, [self.n_users, self.n_items], 0)

        return u_embed, i_embed

    def _create_norm_embed_v2(self):

        A_fold_hat = self._split_A_hat(self.norm_adj_dif)

        ego_embeddings_v = tf.concat([self.um_v2, self.im_v2], axis=0)

        for k in range(0, 1):

            temp_embed = []

            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings_v))

            side_embeddings = tf.concat(temp_embed, 0)

            ego_embeddings_v = side_embeddings

        u_embed, i_embed = tf.split(ego_embeddings_v, [self.n_users, self.n_items], 0)

        return u_embed, i_embed

    def _create_norm_embed_t2(self):

        A_fold_hat = self._split_A_hat(self.norm_adj_dif)

        ego_embeddings_t = tf.concat([self.um_t2, self.im_t2], axis=0)

        for k in range(0, 1):

            temp_embed = []

            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings_t))

            side_embeddings = tf.concat(temp_embed, 0)

            ego_embeddings_t = side_embeddings

        u_embed, i_embed = tf.split(ego_embeddings_t, [self.n_users, self.n_items], 0)

        return u_embed, i_embed

    def _create_norm_embed_m(self):

        A_fold_hat = self._split_A_hat(self.norm_adj_m)

        ego_embeddings_m = tf.concat([self.um_m, self.im_m], axis=0)

        for k in range(0, 1):

            temp_embed = []

            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings_m))

            side_embeddings = tf.concat(temp_embed, 0)

            ego_embeddings_m = side_embeddings

        u_embed, i_embed = tf.split(ego_embeddings_m, [self.n_users, self.n_items], 0)

        return u_embed, i_embed

    def create_bpr_loss_base(self):
        pos_scores = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.pos_i_g_embeddings), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(self.u_g_embeddings, self.neg_i_g_embeddings), axis=1)

        regularizer_mf = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(self.pos_i_g_embeddings_pre) + \
                         tf.nn.l2_loss(self.neg_i_g_embeddings_pre)

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))

        emb_loss = self.decay * regularizer_mf / self.batch_size

        return mf_loss, emb_loss

    def create_bpr_loss_v(self):

        pos_scores_v = tf.reduce_sum(tf.multiply(self.u_g_embeddings_v, self.pos_i_g_embeddings_v), axis=1)
        neg_scores_v = tf.reduce_sum(tf.multiply(self.u_g_embeddings_v, self.neg_i_g_embeddings_v), axis=1)

        regularizer_mf_v = tf.nn.l2_loss(self.u_g_embeddings_v1_pre) + tf.nn.l2_loss(self.pos_i_g_embeddings_v1_pre) + \
                           tf.nn.l2_loss(self.neg_i_g_embeddings_v1_pre) + \
                           tf.nn.l2_loss(self.u_g_embeddings_v2_pre) + tf.nn.l2_loss(self.pos_i_g_embeddings_v2_pre) + \
                           tf.nn.l2_loss(self.neg_i_g_embeddings_v2_pre)

        mf_loss_v = tf.reduce_mean(tf.nn.softplus(-(pos_scores_v - neg_scores_v)))

        emb_loss_v = self.decay * regularizer_mf_v / self.batch_size

        return mf_loss_v, emb_loss_v

    def create_bpr_loss_t(self):

        pos_scores_t = tf.reduce_sum(tf.multiply(self.u_g_embeddings_t, self.pos_i_g_embeddings_t), axis=1)
        neg_scores_t = tf.reduce_sum(tf.multiply(self.u_g_embeddings_t, self.neg_i_g_embeddings_t), axis=1)

        regularizer_mf_t = tf.nn.l2_loss(self.u_g_embeddings_t1_pre) + tf.nn.l2_loss(self.pos_i_g_embeddings_t1_pre) + \
                           tf.nn.l2_loss(self.neg_i_g_embeddings_t1_pre) + \
                           tf.nn.l2_loss(self.u_g_embeddings_t2_pre) + tf.nn.l2_loss(self.pos_i_g_embeddings_t2_pre) + \
                           tf.nn.l2_loss(self.neg_i_g_embeddings_t2_pre)

        mf_loss_t = tf.reduce_mean(tf.nn.softplus(-(pos_scores_t - neg_scores_t)))

        emb_loss_t = self.decay * regularizer_mf_t / self.batch_size

        return mf_loss_t, emb_loss_t

    def create_bpr_loss_m(self):

        pos_scores_m = tf.reduce_sum(tf.multiply(self.u_g_embeddings_m, self.pos_i_g_embeddings_m), axis=1)
        neg_scores_m = tf.reduce_sum(tf.multiply(self.u_g_embeddings_m, self.neg_i_g_embeddings_m), axis=1)

        regularizer_mf_m = tf.nn.l2_loss(self.u_g_embeddings_m) + tf.nn.l2_loss(self.pos_i_g_embeddings_m) + \
                           tf.nn.l2_loss(self.neg_i_g_embeddings_m)

        mf_loss_m = tf.reduce_mean(tf.nn.softplus(-(pos_scores_m - neg_scores_m)))

        emb_loss_m = self.decay * regularizer_mf_m / self.batch_size

        return mf_loss_m, emb_loss_m

    def cosine_similarity(self, embed1, embed2):

        dot_product = tf.reduce_sum(tf.multiply(embed1, embed2), axis=-1)
        norm1 = tf.sqrt(tf.reduce_sum(tf.square(embed1), axis=-1))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(embed2), axis=-1))

        cosine_sim = dot_product / (norm1 * norm2)

        return cosine_sim

    def cos_square_dis(self, x, y):

        x_square = tf.reduce_sum(tf.square(x), axis=1)
        y_square = tf.reduce_sum(tf.square(y), axis=1)
        xy_square = tf.square(tf.reduce_sum(x * y, axis=1))
        cov = tf.reduce_mean(tf.sqrt(xy_square / (x_square * y_square) + 1e-8), axis=0)

        return cov

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    # tf.enable_eager_execution()

    if not os.path.exists('Log/'):
        os.mkdir('Log/')
    file = open('Log/ours-{}-result-{}-decay={}-layer=5.txt'.format(time.time(), dataset, decay), 'a')

    cores = multiprocessing.cpu_count() // 3
    Ks = np.arange(1, 21)

    data_generator.print_statistics()
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['decay'] = decay
    config['n_layers'] = n_layers
    config['embed_size'] = embed_size
    config['lr'] = lr
    config['batch_size'] = batch_size

    """
    ################################################################################
    Generate the Laplacian matrix.
    """
    norm_left, norm_3, norm_4, norm_5 = data_generator.get_adj_mat()

    config['norm_adj'] = norm_3
    config['norm_adj_com'] = norm_5
    config['norm_adj_dif'] = norm_5
    config['norm_adj_m'] = norm_5

    print('shape of adjacency', norm_left.shape)

    t0 = time.time()

    model = Model(data_config=config,
                  img_feat=data_generator.imageFeaMatrix,
                  text_feat=data_generator.textFeatMatrix,
                  d1=4096, d2=300)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver(tf.global_variables())

    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.

    """
    ################################################################################
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False
    max_recall, max_precision, max_ndcg, max_hr = 0., 0., 0., 0.
    max_epoch = 0
    early_stopping = 0

    best_score = 0
    best_result = {}
    all_result = {}

    for epoch in range(500):
        t1 = time.time()
        loss, mf_loss, emb_loss, kd_loss, fd_loss = 0., 0., 0., 0., 0.
        n_batch = data_generator.n_train // batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample_u()

            _, batch_mf_loss, batch_emb_loss, batch_fd_loss = sess.run(
                [model.opt_1, model.mf_loss, model.emb_loss, model.fd_loss],
                feed_dict={model.users: users,
                           model.pos_items: pos_items,
                           model.neg_items: neg_items
                           })
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss
            fd_loss += batch_fd_loss

        if np.isnan(mf_loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        if (epoch + 1) % interval != 0:
            perf_str = 'Epoch {} [{:.1f}s]: train==[{:.5f} + {:.5f}]'.format(
                epoch, time.time() - t1,
                mf_loss, emb_loss)
            print(perf_str)
            continue

        t2 = time.time()
        users_to_test = list(data_generator.test_set.keys())

        result = test(sess, model, users_to_test, data_generator.exist_items, batch_size, cores)
        hr = result['hit_ratio']
        ndcg = result['ndcg']

        score = hr[4] + ndcg[4]
        if score > best_score:
            best_score = score
            best_result['hr'] = [str(i) for i in hr]
            best_result['ndcg'] = [str(i) for i in ndcg]
            print('best result until now: hr@5,10,20={:.4f},{:.4f},{:.4f},ndcg@5,10,20={:.4f},{:.4f},{:.4f}'.format(
                hr[4], hr[9], hr[19], ndcg[4], ndcg[9], ndcg[19]))
            early_stopping = 0
        else:
            early_stopping += 1

        t3 = time.time()

        perf_str = 'Epoch {} [{:1f}s + {:1f}s]: hit@5=[{:5f}],hit@10=[{:5f}],hit@20=[{:5f}],ndcg@5=[{:5f}],ndcg@10=[{:5f}],ndcg@20=[{:5f}]'.format(epoch, t2 - t1, t3 - t2,
                    hr[4], hr[9], hr[19], ndcg[4], ndcg[9], ndcg[19])
        print(perf_str)
        file.write(perf_str + '\n')
        all_result[epoch + 1] = result
        if early_stopping == 5:
            break
    print('###########################################################################################################################')
    best_perf_str = '[{}], best result: hr@5,10,20={},{},{},ndcg@5,10,20={},{},{}'.format(dataset,
        best_result['hr'][4], best_result['hr'][9], best_result['hr'][19], best_result['ndcg'][4],
        best_result['ndcg'][9], best_result['ndcg'][19])
    print(best_perf_str)
    file.write(best_perf_str + '\n')
    file.close()
