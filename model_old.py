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


        self.Plen = 50
        self.batch_size = 64
        self.num_units = 75
        self.root_path = '/search/odin/wenxin/R-Net'
        self.model_name = 'glove.Deep.50Plen.Adadelta.75h.64b'
        self.restore = False

        self.data_path = self.root_path + '/SQuAD/train_flat.json'
        self.word2vec_path = self.root_path + '/../textGAN/word2vec/glove.6B.300d.txt'
        self.word_list_path = self.root_path + '/SQuAD/vocab.txt'
        self.save_path = self.root_path + '/models/'+ self.model_name
        self.eval_path = self.root_path + '/SQuAD/flat_dev.json'
        self.result_path = self.root_path + '/results/glove.50P.GD.0.01/result.txt'
        self.log_path = self.root_path + '/logs/' + self.model_name
        self.epochs = 100
        

def batch_matmul(mat1, mat2):
    return tf.matmul(mat1, mat2)


def map_fn(mat1, mat2):
    # d3 = mat1.get_shape().to_list()[2]
    # d2 = mat1.get_shape().to_list()[1]
    # m1 = tf.reshape(mat1, [-1, d3])
    # h = tf.matmul(m1, mat2)
    # h = tf.reshape(h, shape=[-1, d2, d3])
    # return h
    return tf.map_fn(fn=lambda x: tf.matmul(x, mat2), elems=mat1)


class RocktaschelAttention:
    def __init__(self, memory):
        '''
        :param memory: shape=[b,l,k]
        '''
        self.memory = memory
        self.m_dims = self.memory.get_shape().as_list()
        with tf.variable_scope('RocktaschelAttention') as scope:
            self.W_m = tf.Variable(np.random.randn(self.m_dims[2], self.m_dims[2])/100, name='att_w_m')
            self.W_v = tf.Variable(np.random.randn(self.m_dims[2], self.m_dims[2])/100, name='att_w_v')
            self.W_u = tf.Variable(np.random.randn(self.m_dims[2], self.m_dims[2])/100, name='att_w_u')
            self.e = tf.constant(np.ones(shape=[self.m_dims[1], 1]))
            self.M = map_fn(self.memory, self.W_m)

    def __call__(self, input_u, input_v):
        '''
        :param input_u: shape=[b,1,k]
        :param input_v: shape=[b,1,k]
        :return:
        '''
        with tf.variable_scope('RocktaschelAttention') as scope:
            V = map_fn(input_v, self.W_v)
            U = map_fn(input_u, self.W_u)
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
    def __init__(self, memory):
        '''
        :param memory: shape=[b,l,k]
        '''
        self.memory = memory
        self.m_dims = self.memory.get_shape().as_list()
        with tf.variable_scope('BahdanauAttention') as scope:
            self.W_m = tf.Variable(np.random.randn(self.m_dims[2], self.m_dims[2])/100, name='att_w_m')
            self.W_h = tf.Variable(np.random.randn(self.m_dims[2], self.m_dims[2])/100, name='att_w_h')
            self.e = tf.constant(np.ones(shape=[self.m_dims[1], 1]))
            self.M = map_fn(self.memory, self.W_m)

    def __call__(self, input_h):
        '''
        :param input_h: shape=[b,1,k]
        :return:
        '''
        with tf.variable_scope('BahdanauAttention') as scope:
            H = map_fn(input_h, self.W_h)
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
        self.mode = mode
        self.build_model()
        

    def QP_encoding_layer(self, embeddings, reuse=False):
        with tf.variable_scope('encoding_layer', reuse=reuse) as scope:
            #self.encoding_cell = tf.contrib.rnn.GRUCell(num_units=self.config.num_units)
            # sequence_length = [embeddings.get_shape()[1] for _ in range(self.config.batch_size)]
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoding_cell,
                                                                     cell_bw=self.encoding_cell,
                                                                     inputs=embeddings,
                                                                     # sequence_length=sequence_length,
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
            #self.matching_cell = tf.contrib.rnn.GRUCell(num_units=2 * self.config.num_units)
            attention_mech = RocktaschelAttention(memory=Q_encodings)
            v = tf.constant(np.zeros(shape=[shape[0], 1, shape[2]]))
            P_encodings_as_list = tf.unstack(value=P_encodings, axis=1)

            states = []
            steps = 0
            for u in P_encodings_as_list:
                if steps != 0:
                    scope.reuse_variables()
                steps += 1
                u = tf.reshape(u, shape=[shape[0], 1, shape[2]])
                context, _ = attention_mech(input_u=u, input_v=v)
                current_in = gate(input_u=u, input_c=context)
                # print current_in.get_shape()
                _, v = self.matching_cell(inputs=tf.reshape(current_in, shape=[shape[0], 2 * shape[2]]),
                                          state=tf.reshape(v, shape=[shape[0], shape[2]]))
                v = tf.reshape(v, shape=[shape[0], 1, shape[2]])
                states.append(v)
            stacked_states = tf.concat(states, axis=1)
            print stacked_states.get_shape()
        # stacked_states: [b,plen,k]
        return stacked_states

    def self_matching_layer(self, QP_matchings):
        shape = QP_matchings.get_shape().as_list()
        with tf.variable_scope('self_matching_layer') as scope:
            #self.self_matching_cell = tf.contrib.rnn.GRUCell(num_units=self.config.num_units)
            attention_mech = BahdanauAttention(memory=QP_matchings)
            QP_matchings_as_list = tf.unstack(value=QP_matchings, axis=1)
            att_vec_as_list = []
            steps = 0
            for QP in QP_matchings_as_list:
                if steps!=0:
                    scope.reuse_variables()
                steps+=1
                context, _ = attention_mech(input_h=tf.reshape(QP, shape=[shape[0], 1, shape[2]]))
                context = tf.reshape(context, shape=[shape[0], shape[2]])
                att_vec_as_list.append(tf.concat(values=[context, QP], axis=1))
            att_vec = tf.stack(values=att_vec_as_list, axis=1)
        with tf.variable_scope('self_matching_layer', reuse=False) as scope:
            self_matching, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.self_matching_cell,
                                                                     cell_bw=self.self_matching_cell,
                                                                     inputs=att_vec,
                                                                     # sequence_length=sequence_length,
                                                                     scope=scope,
                                                                     dtype=tf.float64)
            self_matching = tf.concat(self_matching, 2)
        return self_matching

    def boundary_prediction(self, QP_matchings, rQ):
        shape = QP_matchings.get_shape().as_list()
        with tf.variable_scope('boundary_prediction') as scope:
        #with tf.variable_scope('predict_layer') as scope:
            #self.predict_cell = tf.contrib.rnn.GRUCell(num_units=2 * self.config.num_units)
            attention_mech = BahdanauAttention(memory=QP_matchings)
            context_1, score_1 = attention_mech(input_h=rQ)
            h_1, _ = self.predict_cell(inputs=tf.reshape(context_1, shape=[shape[0], shape[2]]),
                                       state=tf.reshape(rQ, shape=[shape[0], shape[2]]))
            scope.reuse_variables()
            context_2, score_2 = attention_mech(input_h=tf.reshape(h_1, shape=[shape[0], 1, shape[2]]))
        return score_1, score_2

    def rQ_layer(self, Q_encodings):
        shape = Q_encodings.get_shape().as_list()
        with tf.variable_scope('rQ_layer'):
            attention_mech = BahdanauAttention(memory=Q_encodings)
            V_Q = tf.constant(np.zeros(shape=[shape[0], 1, shape[2]]))
            context, _ = attention_mech(input_h=V_Q)
        return context

    def build_model(self):
        with tf.variable_scope('self_matching_layer') as scope:
            self.self_matching_cell = tf.contrib.rnn.GRUCell(num_units=self.config.num_units)
        with tf.variable_scope('predict_layer') as scope:
            self.predict_cell = tf.contrib.rnn.GRUCell(num_units=2 * self.config.num_units)        
        with tf.variable_scope('matching_layer') as scope:
            self.matching_cell = tf.contrib.rnn.GRUCell(num_units=2 * self.config.num_units)
        with tf.variable_scope('encoding_layer') as scope:
            self.encoding_cell = tf.contrib.rnn.GRUCell(num_units=self.config.num_units)
        self.buildEmbeddingDict()

        self.Q_ids = tf.placeholder(dtype=tf.int64, shape=[self.config.batch_size, self.config.Qlen])
        self.P_ids = tf.placeholder(dtype=tf.int64, shape=[self.config.batch_size, self.config.Plen])

        Q_embedings = tf.nn.embedding_lookup(params=self.embeddings_map, ids=self.Q_ids)
        P_embedings = tf.nn.embedding_lookup(params=self.embeddings_map, ids=self.P_ids)

        # Encoding layer
        Q_encodings = self.QP_encoding_layer(Q_embedings, reuse=False)
        P_encodings = self.QP_encoding_layer(P_embedings, reuse=True)
        if self.mode=='train':
            Q_encodings = tf.layers.dropout(inputs=Q_encodings, rate=0.15)
            P_encodings = tf.layers.dropout(inputs=P_encodings, rate=0.2)

        # Matching layer, output is paired vectors (att:p_encoding)
        QP_matchings = self.QP_matching_layer(Q_encodings, P_encodings)
        if self.mode == 'train':
            QP_matchings = tf.layers.dropout(inputs=QP_matchings, rate=0.2)

        #Self matching layer
        self_matching = self.self_matching_layer(QP_matchings=QP_matchings)
        if self.mode == 'train':
            self_matching = tf.layers.dropout(inputs=self_matching, rate=0.2)

        rQ = self.rQ_layer(Q_encodings)

        self.score_start, self.score_end = self.boundary_prediction(self_matching, rQ)
        pos_start = tf.arg_max(self.score_start, dimension=1)
        pos_end =tf.arg_max(self.score_end, dimension=1)
        self.pos = tf.stack(values=[pos_start, pos_end], axis=1)
        
        self.var_list = tf.trainable_variables()
        for var in self.var_list:
            print var.name
            tf.summary.histogram(name=var.name,values=var)
    def eval(self, sess):
    	#f=open('/search/odin/wenxin/R-Net/var.txt','r')
    	#lines=f.readlines()
    	#var_dict={}
    	#for i in len(self.var_list):
    	#	var_dict[lines[i]]=self.var_list[i]
        dataloader = DataLoader(self.config.eval_path, batch_size=self.config.batch_size, max_len=self.config.Plen)
        print 'data loaded'
        tf.global_variables_initializer().run()
        print 'variables initialized'
        saver = tf.train.Saver(var_list=self.var_list)
        saver.restore(sess=sess, save_path=self.config.save_path)
        #saver_ = tf.train.Saver(var_list=self.var_list)
        #saver_.save(sess=sess,save_path=self.config.save_path+'.new')
        s = u' '
        epoch = 0
        prediction = dict()
        match_count = 0.
        steps = 0
        total_match = 0.
        while(epoch<1):
            steps+=1
            context_list, question_list, start_label_list, end_label_list, id_list, epoch = dataloader.get_next_batch()
            P_ids_list = self.get_ids_list(context_list, self.config.Plen)
            Q_ids_list = self.get_ids_list(question_list, self.config.Qlen)
            pos_list = sess.run(self.pos, feed_dict={self.P_ids: P_ids_list, self.Q_ids: Q_ids_list})
            pairs = zip(id_list, pos_list.tolist(), context_list, question_list, start_label_list, end_label_list)
            for id, pos, context, _, start, end in pairs:
                answer = s.join(context[pos[0]:(pos[1]+1)])
                prediction[id] = answer
                if pos[0]==start and pos[1]==end:
                    match_count+=1
            if steps%10==0:
                print match_count/(10*self.config.batch_size)
                print steps
                total_match+=match_count
                match_count=0.
            #if steps == 30:
            #    break
        if steps%10!=0:
            total_match+=match_count
        print total_match/(steps*self.config.batch_size)
        f=open(self.config.result_path, 'w')
        f.write(json.dumps(prediction))
        #count = 0
        #for _, pos, context, question, start, end in pairs:
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
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.config.learning_rate, rho=0.95, epsilon=1e-6)#.minimize(loss=self.loss)
        print 'built loss'

        grad_and_vars = optimizer.compute_gradients(loss=self.loss)

        for grad, var in grad_and_vars:
            tf.summary.histogram(name='grad_'+var.name, values=grad)

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
            _, loss, pos_list, summary_str = sess.run([opt, self.loss, self.pos, summary], feed_dict={labels_start: start_label_list,
                                                                                        labels_end: end_label_list,
                                                                                        self.P_ids: P_ids_list,
                                                                                        self.Q_ids: Q_ids_list})
            loss_sum += loss
            summary_writer.add_summary(summary=summary_str, global_step=steps)
            for i in range(self.config.batch_size):
                if start_label_list[i]==pos_list[i][0] and end_label_list[i]==pos_list[i][1]:
                    match_count += 1
            if steps % 10 == 0:
                s = u' '
                context = context_list[0]
                question = question_list[0]
                pos = pos_list.tolist()[0]
                start = start_label_list[0]
                end = end_label_list[0]
                print match_count/(self.config.batch_size*10)
                match_count = 0.
                try:
                    print s.join(context)
                except Exception as e:
                    print context
                try:
                    print s.join(question)
                except Exception as e:
                    print question
                print [start, end]
                print pos
                print 'Ground Truth: ', context[start:end+1]
                try:
                    print 'Eval Answer: ',context[pos[0]:(pos[1]+1)]
                except Exception as e:
                    print 'Out of context range!'
                print 'loss: ',loss_sum / 10
                print 'epoch: ',epoch
                print 'steps: ',steps
                print self.config.model_name
                print '=========================\n'
                self.saver.save(sess=sess,save_path=self.config.save_path)
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
