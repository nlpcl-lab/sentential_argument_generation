import numpy as np
import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope


def attention_decoder(emb_dec_inp, dec_init_state, enc_outputs, enc_pad_masks, cell, initial_state_attention):
    with variable_scope.variable_scope('attention_decoder', reuse=tf.AUTO_REUSE) as scope:
        batch_size = enc_outputs.get_shape()[0].value

        # length of encoded input to attend
        attn_lens = enc_outputs.get_shape()[2].value

        # [batch_size, attn_lens, 1, attn_size)
        enc_outputs = tf.expand_dims(enc_outputs, axis=2)

        W_h = tf.get_variable('W_h', [1, 1, attn_lens, attn_lens])
        enc_features = nn_ops.conv2d(enc_outputs, W_h, [1, 1, 1, 1, ], 'SAME')
        v = tf.get_variable('v', [attn_lens])

        def attention(decoder_state):
            with tf.variable_scope('attention', reuse=tf.AUTO_REUSE) as attn_scope:
                decoder_features = linear(decoder_state, attn_lens, True)
                decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)

                def masked_attention(e):
                    """Take softmax of e then apply enc_padding_mask and re-normalize"""
                    attn_dist = nn_ops.softmax(e)  # take softmax. shape (batch_size, attn_length)
                    attn_dist *= enc_pad_masks  # apply mask
                    masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                    return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize

                # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
                e = math_ops.reduce_sum(
                    v * math_ops.tanh(enc_features + decoder_features), [2, 3])
                # Calculate attention distribution
                attn_dist = masked_attention(e)

                # Calculate the context vector from attn_dist and encoder_states
                context_vector = math_ops.reduce_sum(
                    array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * enc_outputs,
                    [1, 2])  # shape (batch_size, attn_size).
                context_vector = array_ops.reshape(context_vector, [-1, attn_lens])
            return context_vector, attn_dist

        outputs = []
        attn_dists = []
        state = dec_init_state
        context_vector = array_ops.zeros([batch_size, attn_lens])

        # Ensure the second shape of attention vectors is set.
        context_vector.set_shape([None, attn_lens])

        if initial_state_attention:  # true in decode mode
            # Re-calculate the context vector from the previous step
            # so that we can pass it through a linear layer with
            # this step's input to get a modified version of the input
            context_vector, _ = attention(dec_init_state)
        for i, inp in enumerate(emb_dec_inp):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()

            # Merge input and previous attentions into one vector x of the same size as inp
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            x = linear([inp] + [context_vector], input_size, True)

            cell_output, state = cell(x, state)

            # Run the attention mechanism.
            if i == 0 and initial_state_attention:  # always true in decode mode
                with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                                   reuse=tf.AUTO_REUSE):
                    context_vector, attn_dist = attention(state)
            else:
                context_vector, attn_dist = attention(state)

            attn_dists.append(attn_dist)

            # Concatenate the cell_output (= decoder state) and the context vector,
            # and pass them through a linear layer
            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + [context_vector], cell.output_size, True)
            outputs.append(output)

        return outputs, state, attn_dists


def multihead_attention_decoder(emb_dec_inp, dec_init_state, enc_outputs, enc_pad_masks, cell, enc_out_state, multihead_num, initial_state_attention, selected_idx):
    with tf.variable_scope('attention_decoder', reuse=tf.AUTO_REUSE) as scope:
        batch_size = enc_outputs.get_shape()[0].value

        # size of encoded input to attend
        attn_size = enc_outputs.get_shape()[2].value

        # [batch_size, attn_lens, 1, attn_size)
        enc_outputs = tf.expand_dims(enc_outputs, axis=2)

        W_h = tf.get_variable('W_h', [1, 1, attn_size, attn_size * multihead_num])
        enc_features = nn_ops.conv2d(enc_outputs, W_h, [1, 1, 1, 1], 'SAME')

        # [batch_size * multihead_num, attn_lens, 1, attn_size]
        enc_features = tf.reshape(enc_features, [batch_size * multihead_num, -1, 1, attn_size])
        v = tf.get_variable('v', [attn_size])

        def attention(decoder_state):
            with tf.variable_scope('attention', reuse=tf.AUTO_REUSE) as attn_scope:
                decoder_features = linear(decoder_state, attn_size, True)
                decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)

                def masked_attention(e):
                    """Take softmax of e then apply enc_padding_mask and re-normalize"""
                    attn_dist = nn_ops.softmax(e)  # take softmax. shape (batch_size, attn_length)
                    attn_dist *= tf.reshape(tf.tile(enc_pad_masks, [1, multihead_num]),
                                            [batch_size * multihead_num, -1])  # apply mask
                    masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                    res = attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize
                    return res

                # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
                e = math_ops.reduce_sum(
                    v * math_ops.tanh(enc_features + tf.reshape(tf.tile(decoder_features, [1, 1, 1, multihead_num]),
                                                                [batch_size * multihead_num, 1, 1, -1])), [2, 3])

                # Calculate attention distribution
                attn_dist = masked_attention(e)  # [batch_size * multihead_num, enc_lens]

                tiled_enc_output = tf.reshape(tf.tile(enc_outputs, [1, 1, 1, multihead_num]), [batch_size * multihead_num, -1, 1, attn_size])

                # Calculate the context vector from attn_dist and encoder_states
                context_vector = math_ops.reduce_sum(array_ops.reshape(attn_dist, [batch_size * multihead_num, -1, 1, 1]) * tiled_enc_output,[1, 2])  # shape (batch_size, attn_size).

                context_vector = array_ops.reshape(context_vector, [-1, attn_size])

                W_q = tf.get_variable('W_q', shape=[multihead_num, 384], dtype=tf.float32,
                                      initializer=tf.random_uniform_initializer())

                # assert that beam size is 5
                tiled_selected_ids = tf.tile([selected_idx], [batch_size])

                head_weight = tf.cond(pred=tf.equal(selected_idx, -1),
                                      true_fn=lambda :tf.nn.softmax(tf.matmul(enc_out_state, W_q, transpose_b=True) + 1e-12),
                                      false_fn=lambda :tf.one_hot(tiled_selected_ids, multihead_num))
                # if tf.equal(selected_idx,-1):
                #     # [batch_size, head_num]
                #     head_weight = tf.nn.softmax(tf.matmul(enc_out_state, W_q, transpose_b=True) + 1e-12)
                # else:
                #     head_weight = tf.one_hot(tf.tile(selected_idx, [multihead_num]), 5)  # beamsize=5
                head_weight = tf.reshape(head_weight, [-1, 1])

                merged_context_vector = tf.multiply(head_weight, context_vector)
                merged_context_vector = tf.reshape(merged_context_vector, [batch_size, multihead_num, attn_size])
                merged_context_vector = tf.reduce_sum(merged_context_vector, 1)

            return merged_context_vector, attn_dist


        outputs = []
        attn_dists = []
        state = dec_init_state
        context_vector = array_ops.zeros([batch_size, attn_size])

        # Ensure the second shape of attention vectors is set.
        context_vector.set_shape([None, attn_size])

        if initial_state_attention:  # true in decode mode
            # Re-calculate the context vector from the previous step
            # so that we can pass it through a linear layer with
            # this step's input to get a modified version of the input
            context_vector, _ = attention(dec_init_state)
        for i, inp in enumerate(emb_dec_inp):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()

            # Merge input and previous attentions into one vector x of the same size as inp
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            x = linear([inp] + [context_vector], input_size, True)

            cell_output, state = cell(x, state)

            # Run the attention mechanism.
            if i == 0 and initial_state_attention:  # always true in decode mode
                with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                                   reuse=tf.AUTO_REUSE):
                    context_vector, attn_dist = attention(state)
            else:
                context_vector, attn_dist = attention(state)

            attn_dists.append(attn_dist)

            # Concatenate the cell_output (= decoder state) and the context vector,
            # and pass them through a linear layer
            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + [context_vector], cell.output_size, True)
            outputs.append(output)

        return outputs, state, attn_dists


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    if isinstance(args, tuple):
        args = args[1]
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size], initializer=tf.constant_initializer(bias_start))
    return res + bias_term