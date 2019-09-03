import os
import codecs
import numpy as np
import tensorflow as tf
from models.attention import attention_decoder


def make_custom_embedding_matrix(vocab, hps):
    if os.path.exists(hps.custom_embed_path + '.npy'):
        mat = np.load(hps.custom_embed_path + '.npy')
        return tf.Variable(mat, trainable=True, name='embed_matrix')

    emb_mat = load_pretrain_word_embedding_matrix(hps, vocab)
    emb_mat = np.array(emb_mat, dtype=np.float32)
    np.save(hps.custom_embed_path, emb_mat)
    assert os.path.exists(hps.custom_embed_path + '.npy')

    emb_mat = tf.Variable(emb_mat, trainable=True, name='embed_matrix')

    return emb_mat


def load_pretrain_word_embedding_matrix(hps, vocab):
    """
    Return the pretrained glove matrix
    """
    assert os.path.exists(hps.embed_path)
    words = {k:None for k in vocab.words}

    with codecs.getreader('utf-8')(tf.gfile.GFile(hps.embed_path, 'rb')) as f:
        for line in f:
            line = line.strip().split()
            if line[0] in vocab.words:
                vec = [float(var) for var in line[1:]]
                assert len(vec) == hps.embed_dim
                words[line[0]] = vec

    unk_cnt = 0
    for word, vec in words.items():
        if vec is None:
            vec = np.random.normal(size=hps.embed_dim)
            words[word] = vec
            unk_cnt += 1

    print("UNK is {}".format(unk_cnt))
    assert len(list(words.keys())) == len(vocab.words)
    matrix = []
    for word in vocab.words:
        matrix.append(words[word])
    return matrix


class BaseModel:
    def __init__(self, vocab, hps):
        self.hps = hps
        self.vocab = vocab

        self.dec_out_states, self.dec_outputs, self.attn_dists = None, None, None
        self.add_placeholder()
        self.initial_scope = 'models' if self.hps.model != 'mmi_bidi' else 'bw_model'

        self.build_model()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if self.hps.mode in ['train', 'lm_train']:
            self.add_train_op()

        self.saver = tf.train.Saver(max_to_keep=10)
        self.summaries = tf.summary.merge_all()

    def build_model(self):
        print('models build start')

        with tf.variable_scope(self.initial_scope):
            self.rand_unif_init = tf.random_uniform_initializer(-self.hps.rand_unif_init_size, self.hps.rand_unif_init_size, seed=777)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=self.hps.trunc_norm_init_std)

            with tf.variable_scope('embedding'):
                print('add embedding')
                self.add_embedding()

            with tf.variable_scope('encoder'):
                print("add encoding")
                self.add_encoder()
                self.reduce_state()

            with tf.variable_scope('decoder'):
                print('add decoding')
                self.add_decoder()

            with tf.variable_scope('output_projection'):
                self.add_output_projection()

            if self.hps.mode not in ['decode']:
                self.add_loss_calculation()
            else:
                if self.hps.model != 'mmi_bidi':
                    assert len(self.dec_vocab_dists) == 1
                    topk_probs_dec, self.topk_ids = tf.nn.top_k(self.dec_vocab_dists[0], self.hps.batch_size * 2)

                    self.topk_log_probs = tf.log(topk_probs_dec)
                else:
                    self.dec_vocab_dists_stack = tf.stack(self.dec_vocab_scores, axis=1)


    def add_embedding(self):
        self.embedding_matrix = make_custom_embedding_matrix(self.vocab, self.hps)
        self.emb_enc_inp = tf.nn.embedding_lookup(self.embedding_matrix, self.enc_batch)
        self.emb_dec_inp = [tf.nn.embedding_lookup(self.embedding_matrix, x) for x in
                            tf.unstack(self.dec_inp_batch, axis=1)]

    def add_encoder(self):
        cell_fw1 = tf.nn.rnn_cell.GRUCell(self.hps.hidden_dim, kernel_initializer=self.rand_unif_init)
        cell_bw1 = tf.nn.rnn_cell.GRUCell(self.hps.hidden_dim, kernel_initializer=self.rand_unif_init)

        cell_fw1 = tf.contrib.rnn.DropoutWrapper(cell=cell_fw1, input_keep_prob=self.keep_rate)
        cell_bw1 = tf.contrib.rnn.DropoutWrapper(cell=cell_bw1, input_keep_prob=self.keep_rate)

        encoder_outputs, (self.bi_fw_st, self.bi_bw_st) = tf.nn.bidirectional_dynamic_rnn(cell_fw1, cell_bw1, self.emb_enc_inp, dtype=tf.float32,
                                                                              sequence_length=self.enc_lens, swap_memory=True)
        # encoder_outputs: rnn-output of each time step.
        # shape: [2(bi-directional), batch_size, max_enc_len(dynamic), hidden_dim]
        #
        # bi_fw_st (or bi_bw_st): output state of forward (or backward) rnn cell.
        # shape: [batch_size, hidden_dim]

        # [batch_size, max_enc_len, 2 * hidden_dim]
        self.encoder_outputs = tf.concat(axis=2, values=encoder_outputs)

    def reduce_state(self):
        """
        Reduce the bi-directional encoder output state size.
        """
        w_reduce = tf.get_variable('w_reduce', [self.hps.hidden_dim * 2, self.hps.decoder_hidden_dim],
                                   dtype=tf.float32, initializer=self.trunc_norm_init)
        b_reduce = tf.get_variable('b_reduce', [self.hps.decoder_hidden_dim],
                                   dtype=tf.float32, initializer=self.trunc_norm_init)
        concat_h = tf.concat(axis=1, values=[self.bi_bw_st, self.bi_fw_st])

        new_h = tf.nn.relu(tf.matmul(concat_h, w_reduce) + b_reduce)

        # [batch_size, hidden_dim]
        self.dec_init_state = new_h


    def add_output_projection(self):
        with tf.variable_scope('dec_output_projection'):
            w = tf.get_variable('w1', [self.hps.decoder_hidden_dim, len(self.vocab.words)],
                                dtype=tf.float32, initializer=self.trunc_norm_init)
            v = tf.get_variable('v1', [len(self.vocab.words)],
                                dtype=tf.float32, initializer=self.trunc_norm_init)

            dec_flatten = tf.reshape(
                tf.stack(self.dec_outputs), [-1, self.hps.decoder_hidden_dim])

            dec_vocab_scores = tf.nn.xw_plus_b(dec_flatten, w, v)
            dec_vocab_scores = tf.reshape(dec_vocab_scores,
                                          [-1, self.hps.batch_size, len(self.vocab.words)])
            dec_vocab_dists = tf.nn.softmax(dec_vocab_scores)

            self.dec_vocab_scores = tf.unstack(dec_vocab_scores)
            self.dec_vocab_dists = tf.unstack(dec_vocab_dists)

    def get_unfreezed_vars(self):
        all_tvars = tf.trainable_variables()
        tvars = []
        for var in all_tvars:
            if 'dec_output_projection' in var.name: tvars.append(var)
        return tvars

    def add_train_op(self):
        loss_to_minimize = self.loss

        tvars = tf.trainable_variables()
        # unfreezed_tvars = self.get_unfreezed_vars()

        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, self.hps.max_grad_norm)

        # gradients_freeze = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        # grads_freeze, global_norm_freeze = tf.clip_by_global_norm(gradients_freeze, self.hps.max_grad_norm)

        tf.summary.scalar('global_norm', global_norm)
        optimizer = tf.train.AdamOptimizer(self.hps.learning_rate)

        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')
        # self.train_op_freeze = optimizer.apply_gradients(zip(grads_freeze, unfreezed_tvars),
        #                                                  global_step=self.global_step, name='train_step_freeze')

    def add_loss_calculation(self):
        self.seq_loss = tf.contrib.seq2seq.sequence_loss(
            tf.stack(self.dec_vocab_scores, axis=1),
            self.dec_tgt_batch, self.dec_pad_masks)
        # TODO: add more loss in here

        self.loss = self.seq_loss

        tf.summary.scalar('loss_seq', self.seq_loss)
        tf.summary.scalar('loss', self.loss)

    def add_placeholder(self):
        self.enc_batch = tf.placeholder(tf.int32, [self.hps.batch_size, None], name='enc_batch')
        self.enc_lens = tf.placeholder(tf.int32, [self.hps.batch_size], name='enc_lens')
        self.enc_pad_masks = tf.placeholder(tf.float32, [self.hps.batch_size, None], name='enc_pad_masks')

        self.dec_lens = tf.placeholder(tf.int32, [self.hps.batch_size], name='dec_lens')
        self.dec_inp_batch = tf.placeholder(tf.int32, [self.hps.batch_size, self.hps.max_dec_len], name='dec_inp_batch')
        self.dec_tgt_batch = tf.placeholder(tf.int32, [self.hps.batch_size, self.hps.max_dec_len], name='dec_tgt_batch')
        self.dec_pad_masks = tf.placeholder(tf.float32, [self.hps.batch_size, self.hps.max_dec_len], name='dec_pad_masks')
        self.keep_rate = tf.placeholder_with_default(self.hps.keep_rate, (), name='keep_rate')
        self.initial_attention = tf.placeholder_with_default(tf.constant(False), shape=[])
        self.freeze_layer = tf.placeholder_with_default(False, shape=(), name='freezing_layer')

    def run_debug_step(self, batch, sess):
        feeddict = self.make_feeddict(batch)
        to_return = {
            'output': self.encoder_outputs
        }
        return sess.run(to_return, feeddict)

    def run_step(self, batch, sess, is_train=False, freeze_layer=False):
        feeddict = self.make_feeddict(batch)
        to_return = {
            'summaries': self.summaries,
            'loss': self.loss,
            'global_step': self.global_step
        }
        if is_train:
            if freeze_layer:
                feeddict[self.freeze_layer] = freeze_layer
                to_return['train_op'] = self.train_op
                # to_return['train_op'] = self.train_op_freeze
            else:
                to_return['train_op'] = self.train_op

        return sess.run(to_return, feeddict)

    def run_encoder(self, batch, sess):
        feeddict = self.make_feeddict(batch, encoder_only=True)
        feeddict[self.keep_rate] = 1.0
        enc_states, dec_init_state = sess.run([self.encoder_outputs, self.dec_init_state], feeddict)
        dec_init_state = dec_init_state[0]

        return enc_states, dec_init_state

    def add_decoder(self):
        cell_fw1 = tf.nn.rnn_cell.GRUCell(self.hps.decoder_hidden_dim, kernel_initializer=self.rand_unif_init)
        cell_fw1 = tf.contrib.rnn.DropoutWrapper(cell=cell_fw1, input_keep_prob=self.keep_rate)

        self.dec_outputs, dec_out_states, attn_dists = attention_decoder(self.emb_dec_inp, self.dec_init_state,
                                                                              self.encoder_outputs, self.enc_pad_masks,
                                                                              cell_fw1,
                                                                              initial_state_attention=self.hps.mode == 'decode')
        self.dec_out_states = (dec_out_states, [])
        self.attn_dists = (attn_dists, [], [])

    def decode_bidi_seq(self, batch, sess, enc_states, dec_init_states):
        beam_size = len(dec_init_states)

        hiddens = [np.expand_dims(state, axis=0) for state in dec_init_states]
        new_hiddens = np.concatenate(hiddens, axis=0)

        feed = {
            self.encoder_outputs: enc_states,
            self.enc_pad_masks: batch.enc_pad_mask,
            self.dec_init_state: new_hiddens,
            self.dec_inp_batch: batch.dec_inp_batch,
            self.initial_attention: False,
            self.keep_rate: 1.0}

        to_return = {
            'probs': self.dec_vocab_dists_stack
        }

        result = sess.run(to_return, feed_dict=feed)

        return result['probs']

    def decode_onestep(self, batch, sess, latest_tokens, enc_states, dec_init_states, first_step=False, selected_ids=-1):
        beam_size = len(dec_init_states)

        # Concat the list of state into [batch_size, hidden_dim]
        hiddens = [np.expand_dims(state, axis=0) for state in dec_init_states]
        new_hiddens = np.concatenate(hiddens, axis=0)

        feed = {
            self.encoder_outputs: enc_states,
            self.enc_pad_masks: batch.enc_pad_mask,
            self.dec_init_state: new_hiddens,
            self.dec_inp_batch: np.transpose(np.array([latest_tokens])),
            self.initial_attention: not first_step,
            self.keep_rate: 1.0}

        to_return = {
            'ids': self.topk_ids,
            'probs': self.topk_log_probs,
            'last_states': self.dec_out_states[0],
            'attn_dists': self.attn_dists[0]
        }

        result = sess.run(to_return, feed_dict=feed)
        new_states = [result['last_states'][i] for i in range(beam_size)]

        assert len(result['attn_dists']) == 1
        attn_dists = result['attn_dists'][0].tolist()

        return result['ids'], result['probs'], new_states, attn_dists

    def make_feeddict(self, batch, encoder_only=False):
        feed_dict = {}
        feed_dict[self.enc_batch] = batch.enc_batch
        feed_dict[self.enc_lens] = batch.enc_lens
        feed_dict[self.enc_pad_masks] = batch.enc_pad_mask
        feed_dict[self.keep_rate] = self.hps.keep_rate

        if not encoder_only:
            feed_dict[self.dec_inp_batch] = batch.dec_inp_batch
            feed_dict[self.dec_tgt_batch] = batch.dec_tgt_batch
            feed_dict[self.dec_pad_masks] = batch.dec_pad_mask
        return feed_dict
