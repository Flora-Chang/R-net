import tensorflow as tf
import numpy as np
import random
import string
import json
import word2vec.word2vec as word2vec
from dataloader import DataLoader


class CONFIG:
    def __init__(self):
        self.Qlen = 20
        self.embedding_size = 300
        self.learning_rate = 1

        self.Plen = 150
        self.att_size = 75
        self.batch_size = 64
        self.num_units = 75
        self.root_path = '/search/odin/wenxin/R-Net'
        self.model_name ='glove.Deep.150Plen.Adadelta.75h.64b.0.2r' #'glove.50Plen.Adadelta.75h.64b'
        self.restore = False

        self.data_path = self.root_path + '/SQuAD/train_flat.json'
        self.word2vec_path = self.root_path + '/../textGAN/word2vec/glove.6B.300d.txt'
        self.word_list_path = self.root_path + '/SQuAD/vocab.txt'
        self.save_path = self.root_path + '/models/' + self.model_name
        self.eval_path = self.root_path + '/SQuAD/flat_dev.json'
        self.result_path = self.root_path + '/results/glove.50P.GD.0.01/result.txt'
        self.log_path = self.root_path + '/logs/' + self.model_name
        self.epochs = 20


def batch_matmul(mat1, mat2):
    return tf.matmul(mat1, mat2)


def map_fn(mat1, mat2):
    return tf.map_fn(fn=lambda x: tf.matmul(x, mat2), elems=mat1)


class RocktaschelAttention:
    def __init__(self, memory, att_size):
        '''
        :param memory: shape=[b,l,k]
        '''
        self.memory = memory
        self.m_dims = self.memory.get_shape().as_list()
        self.att_size = att_size
        with tf.variable_scope('RocktaschelAttention') as scope:
            W_m = tf.get_variable(name='att_w_m', shape=[self.m_dims[2], att_size], dtype=tf.float64)
            self.e = tf.constant(np.ones(shape=[self.m_dims[1], 1]))
            self.M = map_fn(self.memory, W_m)

    def __call__(self, input_u, input_v):
        '''
        :param input_u: shape=[b,1,k]
        :param input_v: shape=[b,1,k]
        :return:
        '''
        with tf.variable_scope('RocktaschelAttention') as scope:
            W_v = tf.get_variable(name='att_w_v', shape=[input_v.get_shape()[2], self.att_size], dtype=tf.float64)
            W_u = tf.get_variable(name='att_w_u', shape=[input_u.get_shape()[2], self.att_size], dtype=tf.float64)
            V = map_fn(input_v, W_v)
            U = map_fn(input_u, W_u)
            tanh_o = tf.tanh(self.M + V + U)
            score = tf.contrib.layers.fully_connected(inputs=tanh_o,
                                                      num_outputs=1,
                                                      activation_fn=None,
                                                      biases_initializer=None)
            # score: [b,l]
            score = tf.reshape(score, shape=[self.m_dims[0], self.m_dims[1]])
            # alignments: [b,1,l]
            alignments = tf.reshape(tf.nn.softmax(score), shape=[self.m_dims[0], 1, self.m_dims[1]])
            # context: [b,1,k]
            context = batch_matmul(alignments, self.memory)
        return context, score


class BahdanauAttention:
    def __init__(self, memory, att_size):
        '''
        :param memory: shape=[b,l,k]
        '''
        self.memory = memory
        self.m_dims = self.memory.get_shape().as_list()
        self.att_size = att_size
        with tf.variable_scope('BahdanauAttention') as scope:
            W_m = tf.get_variable(name='att_w_m', shape=[self.m_dims[2], att_size], dtype=tf.float64)
            self.e = tf.constant(np.ones(shape=[self.m_dims[1], 1]))
            self.M = map_fn(self.memory, W_m)

    def __call__(self, input_h):
        '''
        :param input_h: shape=[b,1,k]
        :return:
        '''
        with tf.variable_scope('BahdanauAttention') as scope:
            W_h = tf.get_variable(name='att_w_h', shape=[input_h.get_shape()[2], self.att_size], dtype=tf.float64)
            H = map_fn(input_h, W_h)
            tanh_o = tf.tanh(self.M + H)
            score = tf.contrib.layers.fully_connected(inputs=tanh_o,
                                                      num_outputs=1,
                                                      activation_fn=None,
                                                      biases_initializer=None)
            # score: [b,l]
            score = tf.reshape(score, shape=[self.m_dims[0], self.m_dims[1]])
            # alignments: [b,1,l]
            alignments = tf.reshape(tf.nn.softmax(score), shape=[self.m_dims[0], 1, self.m_dims[1]])
            # context: [b,1,k]
            context = batch_matmul(alignments, self.memory)
        return context, score


def gate(input_u, input_c):
    with tf.variable_scope('gate'):
        # concat: [b,1,2k]
        concat = tf.concat(values=[input_u, input_c], axis=2)

        # Here wether the num_outputs is 1 or 2k still remains discussion.
        # For num_outputs=1 the gate calculate an input rate for whole input vectors
        # For num_outputs=2k the gate calculate an input rate element-wise.
        o = tf.contrib.layers.fully_connected(inputs=concat,
                                              num_outputs=1,
                                              activation_fn=None)
        # g: [b,1,1]
        g = tf.nn.sigmoid(o)
        # next_input: [b, 1, 4k]
        next_input = tf.multiply(g, concat)
    return next_input


class RNET:
    def __init__(self, mode='train'):
        self.config = CONFIG()
        if mode == 'train':
            self.mode = True
        else: 
            self.mode = False
        self.build_model()

    def QP_encoding_layer(self, embeddings, reuse=False):
        with tf.variable_scope('encoding_layer', reuse=reuse) as scope:
            encoding_cell_1 = tf.contrib.rnn.GRUCell(num_units=self.config.num_units)
            encoding_cell_2 = tf.contrib.rnn.GRUCell(num_units=self.config.num_units)
            encoding_cell_3 = tf.contrib.rnn.GRUCell(num_units=self.config.num_units)
            self.encoding_cell = tf.contrib.rnn.MultiRNNCell(cells=[encoding_cell_1, encoding_cell_2, encoding_cell_3])
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoding_cell,
                                                                     cell_bw=self.encoding_cell,
                                                                     inputs=embeddings,
                                                                     scope=scope,
                                                                     dtype=tf.float64)
            outputs = tf.concat(outputs, 2)

        return outputs

    def QP_matching_layer(self, Q_encodings, P_encodings):
        """
        Q_encodings: [batch_size, Qlen, 2*embedding_size]
        P_encodings: [batch_size, Plen, 2*embedding_size]
        """
        shape = Q_encodings.get_shape().as_list()
        print shape
        with tf.variable_scope('matching_layer', reuse=False) as scope:
            self.matching_cell = tf.contrib.rnn.GRUCell(num_units=self.config.num_units)
            attention_mech = RocktaschelAttention(memory=Q_encodings, att_size=self.config.att_size)
            P_encodings_as_list = tf.unstack(value=P_encodings, axis=1)

            steps = 0
            states = []
            v = tf.constant(np.zeros(shape=[self.config.batch_size, 1, self.config.num_units]))
            for u in P_encodings_as_list:
                if steps != 0:
                    scope.reuse_variables()
                steps += 1
                u = tf.reshape(u, shape=[shape[0], 1, shape[2]])
                context, _ = attention_mech(input_u=u, input_v=v)
                current_in = gate(input_u=u, input_c=context)
                # print current_in.get_shape()
                _, v = self.matching_cell(inputs=tf.reshape(current_in, shape=[shape[0], 2 * shape[2]]),
                                          state=tf.reshape(v, shape=[self.config.batch_size, self.config.num_units]))
                v = tf.reshape(v, shape=[self.config.batch_size, 1, self.config.num_units])
                states.append(v)
            stacked_states = tf.concat(states, axis=1)
            print stacked_states.get_shape()

        # stacked_states: [b,plen,k]
        return stacked_states

    def self_matching_layer(self, QP_matchings):
        shape = QP_matchings.get_shape().as_list()
        with tf.variable_scope('self_matching_layer') as scope:
            self.self_matching_cell = tf.contrib.rnn.GRUCell(num_units=self.config.num_units)
            attention_mech = BahdanauAttention(memory=QP_matchings, att_size=self.config.att_size)
            QP_matchings_as_list = tf.unstack(value=QP_matchings, axis=1)
            att_vec_as_list = []
            steps = 0
            for QP in QP_matchings_as_list:
                if steps != 0:
                    scope.reuse_variables()
                steps += 1
                context, _ = attention_mech(input_h=tf.reshape(QP, shape=[shape[0], 1, shape[2]]))
                context = tf.reshape(context, shape=[shape[0], shape[2]])
                att_vec_as_list.append(tf.concat(values=[context, QP], axis=1))
            att_vec = tf.stack(values=att_vec_as_list, axis=1)
        with tf.variable_scope('self_matching_layer', reuse=False) as scope:
            self_matching, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.self_matching_cell,
                                                               cell_bw=self.self_matching_cell,
                                                               inputs=att_vec,
                                                               scope=scope,
                                                               dtype=tf.float64)
            self_matching = tf.concat(self_matching, 2)
        return self_matching

    def boundary_prediction(self, QP_matchings, rQ):
        shape = QP_matchings.get_shape().as_list()
        # with tf.variable_scope('boundary_prediction') as scope:
        with tf.variable_scope('predict_layer') as scope:
            self.predict_cell = tf.contrib.rnn.GRUCell(num_units=self.config.num_units)
            attention_mech = BahdanauAttention(memory=QP_matchings, att_size=self.config.att_size)
            context_1, score_1 = attention_mech(input_h=rQ)
            h_1, _ = self.predict_cell(inputs=tf.reshape(context_1, shape=[shape[0], shape[2]]),
                                       state=tf.reshape(rQ, shape=[self.config.batch_size, self.config.num_units]))
            scope.reuse_variables()
            context_2, score_2 = attention_mech(
                input_h=tf.reshape(h_1, shape=[self.config.batch_size, 1, self.config.num_units]))
        return score_1, score_2

    def rQ_layer(self, Q_encodings):
        shape = Q_encodings.get_shape().as_list()
        with tf.variable_scope('rQ_layer'):
            attention_mech = BahdanauAttention(memory=Q_encodings, att_size=self.config.att_size)
            V_Q = tf.constant(np.zeros(shape=[shape[0], 1, shape[2]]))
            context, _ = attention_mech(input_h=V_Q)
            context = tf.contrib.layers.fully_connected(inputs=context,
                                                        num_outputs=self.config.num_units,
                                                        activation_fn=tf.nn.relu)
        return context

    def build_model(self):
        self.buildEmbeddingDict()

        self.Q_ids = tf.placeholder(dtype=tf.int64, shape=[self.config.batch_size, self.config.Qlen])
        self.P_ids = tf.placeholder(dtype=tf.int64, shape=[self.config.batch_size, self.config.Plen])

        Q_embedings = tf.nn.embedding_lookup(params=self.embeddings_map, ids=self.Q_ids)
        P_embedings = tf.nn.embedding_lookup(params=self.embeddings_map, ids=self.P_ids)

        # Encoding layer
        Q_encodings = self.QP_encoding_layer(Q_embedings, reuse=False)
        P_encodings = self.QP_encoding_layer(P_embedings, reuse=True)
        Q_encodings = tf.layers.dropout(inputs=Q_encodings, rate=0.1, training=self.mode)
        P_encodings = tf.layers.dropout(inputs=P_encodings, rate=0.2, training=self.mode)

        # Matching layer, output is paired vectors (att:p_encoding)
        QP_matchings = self.QP_matching_layer(Q_encodings, P_encodings)
        QP_matchings = tf.layers.dropout(inputs=QP_matchings, rate=0.2, training=self.mode)

        # Self matching layer
        self_matching = self.self_matching_layer(QP_matchings=QP_matchings)
        self_matching = tf.layers.dropout(inputs=self_matching, rate=0.2, training=self.mode)

        rQ = self.rQ_layer(Q_encodings)

        self.score_start, self.score_end = self.boundary_prediction(self_matching, rQ)
        self.prob_start = tf.nn.softmax(self.score_start)
        self.prob_end = tf.nn.softmax(self.score_end)
        pos_start = tf.arg_max(self.prob_start, dimension=1)
        pos_end = tf.arg_max(self.prob_end, dimension=1)
        self.pos = tf.stack(values=[pos_start, pos_end], axis=1)
        

        self.var_list = tf.trainable_variables()
        for var in self.var_list:
            print var.name
            tf.summary.histogram(name=var.name, values=var)

    def eval(self, sess):
        dataloader = DataLoader(self.config.eval_path, batch_size=self.config.batch_size, max_len=self.config.Plen)
        print 'data loaded'
        tf.global_variables_initializer().run()
        print 'variables initialized'
        saver = tf.train.Saver(var_list=self.var_list)
        saver.restore(sess=sess, save_path=self.config.save_path)
        s = u' '
        epoch = 0
        prediction = dict()
        match_count = 0.
        steps = 0
        total_match = 0.
        while (epoch < 1):
            steps += 1
            context_list, question_list, start_label_list, end_label_list, id_list, epoch = dataloader.get_next_batch()
            P_ids_list = self.get_ids_list(context_list, self.config.Plen)
            Q_ids_list = self.get_ids_list(question_list, self.config.Qlen)
            pos_list, start, end = sess.run([self.pos, self.prob_start, self.prob_end], feed_dict={self.P_ids: P_ids_list, 
                                                                                                     self.Q_ids: Q_ids_list})
            pos_list = zip(start.tolist(), end.tolist())
            pairs = zip(id_list, pos_list, context_list, question_list, start_label_list, end_label_list)
            for id, pos, context, _, start, end in pairs:
                l_s = pos[0]
                l_e = pos[1]
                pos_s = 0
                pos_e = 0
                if max(l_s)>=max(l_e):
                    pos_s=np.argmax(l_s)
                    pos_e=np.argmax(l_e[np.argmax(l_s):])+pos_s
                else:
                    pos_e=np.argmax(l_e)
                    pos_s=np.argmax(l_s[0:np.argmax(l_e)+1])
                pos = [pos_s, pos_e]
                answer = s.join(context[pos[0]:(pos[1] + 1)])
                prediction[id] = answer
                if pos[0] == start and pos[1] == end:
                    match_count += 1
            if steps % 10 == 0:
                print match_count / (10 * self.config.batch_size)
                print steps
                total_match += match_count
                match_count = 0.
                # if steps == 30:
                #    break
        if steps % 10 != 0:
            total_match += match_count
        print total_match / (steps * self.config.batch_size)
        f = open(self.config.result_path, 'w')
        f.write(json.dumps(prediction))
        # count = 0
        # for _, pos, context, question, start, end in pairs:
        #    try:
        #        print s.join(context)
        #    except Exception as e:
        #        print context
        #    print s.join(question)
        #    print context[start:(end+1)]
        #    print context[pos[0]:(pos[1]+1)]
        #    print '==============================\n'
        #    count+=1
        #    if count == 20:
        #        break

    def train(self, sess):
        labels_start = tf.placeholder(dtype=tf.int64, shape=[self.config.batch_size])
        labels_end = tf.placeholder(dtype=tf.int64, shape=[self.config.batch_size])

        logits = tf.concat(values=[self.score_start, self.score_end], axis=0)
        labels = tf.concat(values=[labels_start, labels_end], axis=0)

        print 'build loss'
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.config.learning_rate, rho=0.95,
                                               epsilon=1e-6)  # .minimize(loss=self.loss)
        print 'built loss'

        grad_and_vars = optimizer.compute_gradients(loss=self.loss)

        for grad, var in grad_and_vars:
            tf.summary.histogram(name='grad_' + var.name, values=grad)

        opt = optimizer.apply_gradients(grads_and_vars=grad_and_vars)
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(logdir=self.config.log_path)
        dataloader = DataLoader(self.config.data_path, batch_size=self.config.batch_size, max_len=self.config.Plen)
        print 'data loaded'
        tf.global_variables_initializer().run()
        print 'variables initialized'

        self.saver = tf.train.Saver()
        if self.config.restore:
            self.saver.restore(sess=sess, save_path=self.config.save_path)
        epoch = 0
        loss_sum = 0.
        steps = 0
        match_count = 0.
        while (epoch < self.config.epochs):
            steps += 1
            context_list, question_list, start_label_list, end_label_list, _, epoch = dataloader.get_next_batch()
            P_ids_list = self.get_ids_list(context_list, self.config.Plen)
            Q_ids_list = self.get_ids_list(question_list, self.config.Qlen)
            _, loss, pos_list, summary_str = sess.run([opt, self.loss, self.pos, summary],
                                                      feed_dict={labels_start: start_label_list,
                                                                 labels_end: end_label_list,
                                                                 self.P_ids: P_ids_list,
                                                                 self.Q_ids: Q_ids_list})
            loss_sum += loss
            summary_writer.add_summary(summary=summary_str, global_step=steps)
            for i in range(self.config.batch_size):
                if start_label_list[i] == pos_list[i][0] and end_label_list[i] == pos_list[i][1]:
                    match_count += 1
            if steps % 10 == 0:
                s = u' '
                context = ''
                question = ''
                for i in P_ids_list[0]:
                    context += self.id_vocab_dict[i] + ' '
                for i in Q_ids_list[0]:
                    question += self.id_vocab_dict[i] + ' '
                pos = pos_list.tolist()[0]
                start = start_label_list[0]
                end = end_label_list[0]
                print match_count / (self.config.batch_size * 10)
                match_count = 0.
                print self.config.model_name
                print context
                print context_list[0]
                print question
                print [start, end]
                print pos
                print 'Ground Truth: ', context[start:end + 1]
                try:
                    print 'Eval Answer: ', context[pos[0]:(pos[1] + 1)]
                except Exception as e:
                    print 'Out of context range!'
                print 'loss: ', loss_sum / 10
                print 'epoch: ', epoch
                print 'steps: ', steps
                print '=========================\n'
                self.saver.save(sess=sess, save_path=self.config.save_path)
                loss_sum = 0.

    def get_ids_list(self, context_list, max_len):
        ids_list = []
        for context in context_list:
            ids = []
            i = 0
            for word in context:
                if i == max_len:
                    break
                i += 1
                ids.append(self.vocab_id_dict[word]
                           if self.vocab_id_dict.has_key(word) else self.vocab_id_dict['<unk>'])
            while (i < max_len):
                ids.append(self.vocab_id_dict['<unk>'])
                i += 1
            ids_list.append(ids)
        return ids_list

    def buildEmbeddingDict(self):
        # load word embedding map
        embeddings_dict = word2vec.word2Vec_load(self.config.word2vec_path)
        self.config.embedding_size = len(embeddings_dict['the'])
        f = open(self.config.word_list_path, 'r')
        lines = f.readlines()
        f.close()
        self.vocab_id_dict = dict()
        self.id_vocab_dict = dict()
        self.embeddings_map = []

        def getEmbeddingAsList(embeddings_dict, word):
            if embeddings_dict.has_key(word):
                return embeddings_dict[word]
            else:
                return [0. for _ in range(self.config.embedding_size)]  # embeddings_dict['<unk>']

        i = 0
        for line in lines:
            word = line.strip().split()[0]
            if embeddings_dict.has_key(word):
                self.embeddings_map.append(getEmbeddingAsList(embeddings_dict, word))
                self.vocab_id_dict[word] = i
                self.id_vocab_dict[i] = word
                i += 1
        special_word = ['<unk>']
        for word in special_word:
            self.vocab_id_dict[word] = i
            self.id_vocab_dict[i] = word
            self.embeddings_map.append(getEmbeddingAsList(embeddings_dict, word))
            i += 1
        self.vocab_size = len(self.id_vocab_dict)
        print self.vocab_size
        self.embeddings_map = tf.constant(self.embeddings_map, dtype=tf.float64)
