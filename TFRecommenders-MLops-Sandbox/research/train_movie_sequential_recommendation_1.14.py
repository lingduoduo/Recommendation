# Session-based Recommendations with Recurrent Neural Networks
import argparse
import os
import random
import sys

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

sys.path.append("..")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="DeepRec")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--emb_size", type=int, default=50)
    parser.add_argument("--len_Seq", type=int, default=5)
    parser.add_argument("--len_Tag", type=int, default=1)
    parser.add_argument("--len_Pred", type=int, default=1)
    parser.add_argument("--neg_sample", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--keep_prob", type=float, default=0.7)
    parser.add_argument("--loss_fun", type=str, default="top1")
    parser.add_argument("--l2_lambda", type=float, default=1e-6)
    return parser.parse_args()


# class GRU4Rec(object):
#
# 	def __init__(self, emb_size, num_usr, num_item, len_Seq, len_Tag, layers = 1, loss_fun = 'BPR', l2_lambda = 0.0):
#
# 		self.emb_size = emb_size
# 		self.item_count = num_item
# 		self.user_count = num_usr
# 		self.l2_lambda = l2_lambda
# 		self.layers = layers
# 		self.len_Seq = len_Seq
# 		self.len_Tag = len_Tag
# 		self.loss_fun = loss_fun
#
# 		tf.compat.v1.disable_eager_execution()
# 		self.input_Seq = tf.compat.v1.placeholder(tf.int32, [None, self.len_Seq])  # [B,T]
# 		self.input_NegT = tf.compat.v1.placeholder(tf.int32, [None, None])  # [B,F]
# 		self.input_PosT = tf.compat.v1.placeholder(tf.int32, [None, None])  # [B]
#
# 		self.input_keepprob = tf.compat.v1.placeholder(tf.float32, name = 'keep_prob')
# 		self.loss, self.output = self.build_model(self.input_Seq, self.input_NegT, self.input_PosT, self.input_keepprob)
#
# 	def build_model(self, in_Seq, in_Neg, in_Pos, in_KP):
# 		with tf.compat.v1.variable_scope('gru4rec'):
# 			# Embedding
# 			self.item_emb = tf.compat.v1.get_variable("item_emb", [self.item_count, self.emb_size])  # [N,e]
#
# 			self.W = tf.compat.v1.get_variable("W", [self.item_count, self.emb_size])  # [N,e]
# 			self.b = tf.compat.v1.get_variable("b", [self.item_count, 1])
#
# 			session = tf.nn.embedding_lookup(self.item_emb, in_Seq)  # [B,T,e]
#
# 			cells = []
# 			for _ in range(self.layers):
# 				cell = tf.keras.layers.GRUCell(self.emb_size, activation = tf.nn.tanh)
# 				cell = tf.nn.RNNCellDropoutWrapper(cell, output_keep_prob = in_KP)
# 				cells.append(cell)
# 			self.cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)
#
# 			zero_state = self.cell.zero_state(tf.shape(session)[0], dtype = tf.float32)
#
# 			outputs, state = tf.nn.dynamic_rnn(self.cell, session, initial_state = zero_state)
# 			output = outputs[:, -1:, :]
#
# 			pos_W = tf.nn.embedding_lookup(self.W, in_Pos)
# 			pos_b = tf.nn.embedding_lookup(self.b, in_Pos)
#
# 			neg_W = tf.nn.embedding_lookup(self.W, in_Neg)
# 			neg_b = tf.nn.embedding_lookup(self.b, in_Neg)
#
# 			pos_y = tf.matmul(output, tf.transpose(pos_W, [0, 2, 1])) + tf.transpose(pos_b, [0, 2, 1])
# 			neg_y = tf.matmul(output, tf.transpose(neg_W, [0, 2, 1])) + tf.transpose(neg_b, [0, 2, 1])
#
# 			if self.loss_fun == 'BPR':
# 				loss = self.loss_BPR(pos_y, neg_y)
# 			else:
# 				loss = self.loss_TOP1(pos_y, neg_y)
# 			return loss, output
#
# 	def loss_BPR(self, pos, neg):
# 		Ls = -1 * tf.reduce_mean(tf.compat.v1.log(tf.sigmoid(pos - neg)), -1)
# 		return Ls
#
# 	def loss_TOP1(self, pos, neg):
# 		Ls = tf.reduce_mean(tf.sigmoid(neg - pos) + tf.sigmoid(neg ** 2), -1)
# 		return Ls
#
# 	def predict(self):
# 		score = tf.matmul(tf.squeeze(self.output), tf.transpose(self.W, [1, 0])) + tf.transpose(self.b, [1, 0])
# 		score = tf.sigmoid(score)
# 		return score


class GRU4Rec(object):
    def __init__(
        self,
        emb_size,
        num_usr,
        num_item,
        len_Seq,
        len_Tag,
        layers=1,
        loss_fun="BPR",
        l2_lambda=0.0,
    ):

        self.emb_size = emb_size
        self.item_count = num_item
        self.user_count = num_usr
        self.l2_lambda = l2_lambda
        self.layers = layers
        self.len_Seq = len_Seq
        self.len_Tag = len_Tag
        self.loss_fun = loss_fun

        self.input_Seq = tf.placeholder(tf.int32, [None, self.len_Seq])  # [B,T]
        self.input_NegT = tf.placeholder(tf.int32, [None, None])  # [B,F]
        self.input_PosT = tf.placeholder(tf.int32, [None, None])  # [B]

        self.input_keepprob = tf.placeholder(tf.float32, name="keep_prob")
        self.loss, self.output = self.build_model(
            self.input_Seq, self.input_NegT, self.input_PosT, self.input_keepprob
        )

    def build_model(self, in_Seq, in_Neg, in_Pos, in_KP):
        with tf.variable_scope("gru4rec"):
            # Embedding
            self.item_emb = tf.get_variable(
                "item_emb", [self.item_count, self.emb_size]
            )  # [N,e]

            self.W = tf.get_variable("W", [self.item_count, self.emb_size])  # [N,e]
            self.b = tf.get_variable("b", [self.item_count, 1])

            session = tf.nn.embedding_lookup(self.item_emb, in_Seq)  # [B,T,e]

            cells = []
            for _ in range(self.layers):
                cell = rnn.GRUCell(self.emb_size, activation=tf.nn.tanh)
                cell = rnn.DropoutWrapper(cell, output_keep_prob=in_KP)
                cells.append(cell)
            self.cell = rnn.MultiRNNCell(cells)

            zero_state = self.cell.zero_state(tf.shape(session)[0], dtype=tf.float32)

            outputs, state = tf.nn.dynamic_rnn(
                self.cell, session, initial_state=zero_state
            )
            output = outputs[:, -1:, :]

            pos_W = tf.nn.embedding_lookup(self.W, in_Pos)
            pos_b = tf.nn.embedding_lookup(self.b, in_Pos)

            neg_W = tf.nn.embedding_lookup(self.W, in_Neg)
            neg_b = tf.nn.embedding_lookup(self.b, in_Neg)

            pos_y = tf.matmul(output, tf.transpose(pos_W, [0, 2, 1])) + tf.transpose(
                pos_b, [0, 2, 1]
            )
            neg_y = tf.matmul(output, tf.transpose(neg_W, [0, 2, 1])) + tf.transpose(
                neg_b, [0, 2, 1]
            )

            if self.loss_fun == "BPR":
                loss = self.loss_BPR(pos_y, neg_y)
            else:
                loss = self.loss_TOP1(pos_y, neg_y)
            return loss, output

    def loss_BPR(self, pos, neg):
        Ls = -1 * tf.reduce_mean(tf.log(tf.sigmoid(pos - neg)), -1)
        return Ls

    def loss_TOP1(self, pos, neg):
        Ls = tf.reduce_mean(tf.sigmoid(neg - pos) + tf.sigmoid(neg**2), -1)
        return Ls

    def predict(self):
        score = tf.matmul(
            tf.squeeze(self.output), tf.transpose(self.W, [1, 0])
        ) + tf.transpose(self.b, [1, 0])
        score = tf.sigmoid(score)
        return score


def make_datasets(data, len_Seq, len_Tag, len_Pred):
    p = (
        data.groupby("item")["user"]
        .count()
        .reset_index()
        .rename(columns={"user": "item_count"})
    )
    data = pd.merge(data, p, how="left", on="item")
    data = data[data["item_count"] > 0].drop(["item_count"], axis=1)

    # ReMap item ids
    item_unique = data["item"].unique().tolist()
    item_map = dict(zip(item_unique, range(1, len(item_unique) + 1)))
    item_map[-1] = 0
    all_item_count = len(item_map)
    data["item"] = data["item"].apply(lambda x: item_map[x])

    # ReMap usr ids
    user_unique = data["user"].unique().tolist()
    user_map = dict(zip(user_unique, range(1, len(user_unique) + 1)))
    user_map[-1] = 0
    all_user_count = len(item_map)
    data["user"] = data["user"].apply(lambda x: user_map[x])

    # Get user session
    data = data.sort_values(by=["user", "timestamps"]).reset_index(drop=True)

    # 生成用户序列
    user_sessions = (
        data.groupby("user")["item"]
        .apply(lambda x: x.tolist())
        .reset_index()
        .rename(columns={"item": "item_list"})
    )

    train_users = []
    train_seqs = []
    train_targets = []

    test_users = []
    test_seqs = []
    test_targets = []

    items_usr_clicked = {}

    for index, row in user_sessions.iterrows():
        user = row["user"]
        items = row["item_list"]

        test_item = items[-1 * len_Pred :]
        test_seq = items[-1 * (len_Pred + len_Seq) : -1 * len_Pred]
        test_users.append(user)
        test_seqs.append(test_seq)
        test_targets.append(test_item)

        train_build_items = items[: -1 * len_Pred]

        items_usr_clicked[user] = train_build_items

        for i in range(len_Seq, len(train_build_items) - len_Tag + 1):
            item = train_build_items[i : i + len_Tag]
            seq = train_build_items[max(0, i - len_Seq) : i]

            train_users.append(user)
            train_seqs.append(seq)
            train_targets.append(item)

    d_train = pd.DataFrame(
        {"user": train_users, "seq": train_seqs, "target": train_targets}
    )
    d_test = pd.DataFrame(
        {"user": test_users, "seq": test_seqs, "target": test_targets}
    )
    d_info = (all_user_count, all_item_count, items_usr_clicked, user_map, item_map)

    return d_train, d_test, d_info


class DataIterator:
    def __init__(
        self,
        mode,
        data,
        batch_size=128,
        neg_sample=1,
        all_items=None,
        items_usr_clicked=None,
        shuffle=True,
    ):
        self.mode = mode
        self.data = data
        self.datasize = data.shape[0]
        self.neg_count = neg_sample
        self.batch_size = batch_size
        self.item_usr_clicked = items_usr_clicked
        self.all_items = all_items
        self.shuffle = shuffle
        self.seed = 0
        self.idx = 0
        self.total_batch = round(self.datasize / float(self.batch_size))

    def __iter__(self):
        return self

    def reset(self):
        self.idx = 0
        if self.shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
            self.seed = self.seed + 1
            random.seed(self.seed)

    def __next__(self):

        if self.idx >= self.datasize:
            self.reset()
            raise StopIteration

        nums = self.batch_size
        if self.datasize - self.idx < self.batch_size:
            nums = self.datasize - self.idx

        cur = self.data.iloc[self.idx : self.idx + nums]

        batch_user = cur["user"].values

        batch_seq = []
        for seq in cur["seq"].values:
            batch_seq.append(seq)

        batch_pos = []
        for t in cur["target"].values:
            batch_pos.append(t)

        batch_neg = []
        if self.mode == "train":
            for u in cur["user"]:
                user_item_set = set(self.all_items) - set(self.item_usr_clicked[u])
                batch_neg.append(random.sample(user_item_set, self.neg_count))

        self.idx += self.batch_size

        return (batch_user, batch_seq, batch_pos, batch_neg)


def Metric_HR(TopN, target_list, predict_list):
    sums = 0
    count = 0
    for i in range(len(target_list)):
        preds = predict_list[i]
        top_preds = preds[:TopN]

        for target in target_list[i]:
            if target in top_preds:
                sums += 1
            count += 1

    return float(sums) / count


def Metric_MRR(target_list, predict_list):
    sums = 0
    count = 0
    for i in range(len(target_list)):
        preds = predict_list[i]
        for t in target_list[i]:
            rank = preds.index(t) + 1
            sums += 1 / rank
            count += 1
    return float(sums) / count


if __name__ == "__main__":
    # Get Params
    args = parse_args()
    len_Seq = args.len_Seq  # 序列的长度
    len_Tag = args.len_Tag  # 训练时目标的长度
    len_Pred = args.len_Pred  # 预测时目标的长度
    batch_size = args.batch_size
    emb_size = args.emb_size
    neg_sample = args.neg_sample
    keep_prob = args.keep_prob
    layers = args.layers
    loss_fun = args.loss_fun
    l2_lambda = args.l2_lambda
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    # make datasets

    print("==> make datasets <==")
    file_path = "../data/ml_sample.txt"
    # file_path = 'data/ml_sample.txt'
    names = ["user", "item", "rateing", "timestamps"]
    data = pd.read_csv(file_path, header=None, sep="::", names=names)
    print(data)

    d_train, d_test, d_info = make_datasets(data, len_Seq, len_Tag, len_Pred)
    num_usr, num_item, items_usr_clicked, _, _ = d_info
    all_items = [i for i in range(num_item)]
    print(all_items)

    # Define DataIterator

    trainIterator = DataIterator(
        "train",
        d_train,
        batch_size,
        neg_sample,
        all_items,
        items_usr_clicked,
        shuffle=True,
    )
    testIterator = DataIterator("test", d_test, batch_size, shuffle=False)

    # Define Model

    model = GRU4Rec(emb_size, num_usr, num_item, len_Seq, 1, layers)
    loss = model.loss
    input_Seq = model.input_Seq
    input_NegT = model.input_NegT
    input_PosT = model.input_PosT
    input_keepprob = model.input_keepprob
    score_pred = model.predict()

    # Define Optimizer

    global_step = tf.Variable(0, trainable=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        grads_and_vars = tuple(zip(grads, tvars))
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Training and test for every epoch

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):

            # train

            cost_list = []
            for train_input in tqdm(
                trainIterator,
                desc="epoch {}".format(epoch),
                total=trainIterator.total_batch,
            ):
                batch_usr, batch_seq, batch_pos, batch_neg = train_input
                feed_dict = {
                    input_Seq: batch_seq,
                    input_PosT: batch_pos,
                    input_NegT: batch_neg,
                    input_keepprob: keep_prob,
                }
                _, step, cost = sess.run([train_op, global_step, loss], feed_dict)
                cost_list += list(cost)
            mean_cost = np.mean(cost_list)
            # saver.save(sess, FLAGS.save_path)

            # test

            pred_list = []
            next_list = []
            user_list = []

            for test_input in testIterator:
                batch_usr, batch_seq, batch_pos, batch_neg = test_input
                feed_dict = {input_Seq: batch_seq, input_keepprob: 1.0}
                pred = sess.run(
                    score_pred, feed_dict
                )  # , options=options, run_metadata=run_metadata)

                pred_list += pred.tolist()
                next_list += list(batch_pos)
                user_list += list(batch_usr)

            sorted_items, sorted_score = SortItemsbyScore(
                all_items,
                pred_list,
                reverse=True,
                remove_hist=True,
                usr=user_list,
                usrclick=items_usr_clicked,
            )
            #
            hr50 = Metric_HR(50, next_list, sorted_items)
            Mrr = Metric_MRR(next_list, sorted_items)
            print(
                "epoch {}, mean_loss{:g}, test HR@50: {:g} MRR: {:g}".format(
                    epoch + 1, mean_cost, hr50, Mrr
                )
            )
