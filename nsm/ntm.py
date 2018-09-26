"""Get the data"""

import os
import numpy as np
import math
import tensorflow as tf
import pickle

def load_program_lookup_table(path):
    input_file = os.path.join(path)
    data_dict = {}
    with open(input_file, 'r') as f:
        data = f.read()[1:-1]
    data_list = data.split(',')
    for i in xrange(len(data_list)):
        data_dict[i] = data_list[i].split(':')[1].strip()[1:-1]
    data_dict = {v_i: v for v, v_i in data_dict.items()}
    return data_dict

def load_question_lookup_table(path):
    input_file = os.path.join(path)
    data_dict = {}
    with open(input_file, 'r') as f:
        data = f.read()[1:-1]
    data_list = data.split(', ')

    for i in xrange(len(data_list)):
        start_index = data_list[i].find(': u')
        data_dict[i] = data_list[i][start_index+1:].strip()[2:-1]
    data_dict_new = {}
    for v, v_i in data_dict.items():
        data_dict_new[v_i]=v
    return data_dict_new

# Build the neural network
def enc_dec_model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')

    target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_length)

    return inputs, targets, target_sequence_length, max_target_len


def hyperparam_inputs():
    lr_rate = tf.placeholder(tf.float32, name='lr_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return lr_rate, keep_prob

# add <GO> in each target sentence
# target_data ids
def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for encoding
    :return: Preprocessed target data
    """
    # get '<GO>' id
    go_id = target_vocab_to_int['<GO>']

    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    after_concat = tf.concat([tf.fill([batch_size, 1], go_id), after_slice], 1)

    return after_concat


def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,
                   source_vocab_size,
                   encoding_embedding_size):
    """
    :return: tuple (RNN output, RNN state)
    """
    embed = tf.contrib.layers.embed_sequence(rnn_inputs,
                                             vocab_size=source_vocab_size,
                                             embed_dim=encoding_embedding_size)

    stacked_cells = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size), keep_prob) for _ in range(num_layers)])

    outputs, state = tf.nn.dynamic_rnn(stacked_cells,
                                       embed,
                                       dtype=tf.float32)
    return outputs, state


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                         target_sequence_length, max_summary_length,
                         output_layer, keep_prob):
    """
    Create a training process in decoding layer
    :return: BasicDecoderOutput containing training logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,
                                             output_keep_prob=keep_prob)

    # for only input layer
    helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input,
                                               target_sequence_length)

    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                              helper,
                                              encoder_state,
                                              output_layer)

    # unrolling the decoder layer
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      impute_finished=True,
                                                      maximum_iterations=max_summary_length)
    return outputs


def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    """
    Create a inference process in decoding layer
    :return: BasicDecoderOutput containing inference logits and sample_id
    """
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell,
                                             output_keep_prob=keep_prob)

    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                                      tf.fill([batch_size], start_of_sequence_id),
                                                      end_of_sequence_id)

    decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                              helper,
                                              encoder_state,
                                              output_layer)

    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      impute_finished=True,
                                                      maximum_iterations=max_target_sequence_length)
    return outputs


def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):
    """
    Create decoding layer
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    target_vocab_size = len(target_vocab_to_int)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_layers)])

    with tf.variable_scope("decode1"):
        output_layer = tf.layers.Dense(target_vocab_size)
        train_output = decoding_layer_train(encoder_state,
                                            cells,
                                            dec_embed_input,
                                            target_sequence_length,
                                            max_target_sequence_length,
                                            output_layer,
                                            keep_prob)

    with tf.variable_scope("decode1", reuse=True):
        infer_output = decoding_layer_infer(encoder_state,
                                            cells,
                                            dec_embeddings,
                                            target_vocab_to_int['<GO>'],
                                            target_vocab_to_int['<EOS>'],
                                            max_target_sequence_length,
                                            target_vocab_size,
                                            output_layer,
                                            batch_size,
                                            keep_prob)

    return (train_output, infer_output)


def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence model
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    enc_outputs, enc_states = encoding_layer(input_data,
                                             rnn_size,
                                             num_layers,
                                             keep_prob,
                                             source_vocab_size,
                                             enc_embedding_size)

    dec_input = process_decoder_input(target_data,
                                      target_vocab_to_int,
                                      batch_size)

    train_output, infer_output = decoding_layer(dec_input,
                                                enc_states,
                                                target_sequence_length,
                                                max_target_sentence_length,
                                                rnn_size,
                                                num_layers,
                                                target_vocab_to_int,
                                                target_vocab_size,
                                                batch_size,
                                                keep_prob,
                                                dec_embedding_size)


    return train_output, infer_output


"""Neural Network Training
    hyperparameters
"""

#display_step = 6

epochs = 1
batch_size = 2

rnn_size = 128
num_layers = 3

encoding_embedding_size = 200
decoding_embedding_size = 200

learning_rate = 0.001
keep_probability = 0.5

"""Build the Graph"""
save_path = 'checkpoints/dev'
program_int_to_vocab = '../pre-data/program_int_to_vocab'
question_int_to_vocab = '../pre-data/question_int_to_vocab'
source_vocab_to_int = load_program_lookup_table(program_int_to_vocab)
target_vocab_to_int = load_question_lookup_table(question_int_to_vocab)


train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, target_sequence_length, max_target_sequence_length = enc_dec_model_inputs()
    lr, keep_prob = hyperparam_inputs()

    train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                   targets,
                                                   keep_prob,
                                                   batch_size,
                                                   target_sequence_length,
                                                   max_target_sequence_length,
                                                   len(source_vocab_to_int),
                                                   len(target_vocab_to_int),
                                                   encoding_embedding_size,
                                                   decoding_embedding_size,
                                                   rnn_size,
                                                   num_layers,
                                                   target_vocab_to_int)

    training_logits = tf.identity(train_logits.rnn_output, name='logits1')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions1')

    # https://www.tensorflow.org/api_docs/python/tf/sequence_mask
    # - Returns a mask tensor representing the first N positions of each cell.
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks1')

    with tf.name_scope("optimization1"):
        # Loss function - weighted softmax cross entropy
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks, name='cost1')
        tf.add_to_collection("cost1", cost)
        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients, name='optimizer1')
        tf.add_to_collection("optimizer1", train_op)



def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths


# def get_accuracy(target, logits):
#
#     """
#     Calculate accuracy
#     """
#     max_seq = max(target.shape[1], logits.shape[1])
#     if max_seq - target.shape[1]:
#         target = np.pad(
#             target,
#             [(0,0),(0,max_seq - target.shape[1])],
#             'constant')
#     if max_seq - logits.shape[1]:
#         logits = np.pad(
#             logits,
#             [(0,0),(0,max_seq - logits.shape[1])],
#             'constant')
#
#     return np.mean(np.equal(target, logits))

# Split data to training and validation sets

save_path = '/Users/qingwang/PycharmProjects/ntm_back_translation/checkpoints_back_translation/dev'

# save parameters
def save_params(params):
    with open('/Users/qingwang/PycharmProjects/ntm_back_translation/checkpoints_back_translation/params.p', 'wb') as out_file:
        pickle.dump(params, out_file)

def load_params():
    with open('/Users/qingwang/PycharmProjects/ntm_back_translation/checkpoints_back_translation/params.p', mode='rb') as in_file:
        return pickle.load(in_file)


def back_translation_init(source_int_text, target_int_text):
    train_source = source_int_text
    train_target = target_int_text
    loss = 0

    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epochs):
            for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                    get_batches(train_source, train_target, batch_size,
                                source_vocab_to_int['<PAD>'],
                                target_vocab_to_int['<PAD>'])):
                _, loss = sess.run(
                    [train_op, cost],
                    {input_data: source_batch,
                     targets: target_batch,
                     lr: learning_rate,
                     target_sequence_length: targets_lengths,
                     keep_prob: keep_probability})

                print('Epoch {:>3} Batch {:>4}/{} - Loss: {}'
                      .format(epoch_i, batch_i, len(source_int_text) // batch_size, loss))

        saver = tf.train.Saver()
        saver.save(sess, save_path)
        save_params(save_path)
        return loss * 1.0/targets_lengths[0]


def back_translation_reload(source_int_text, target_int_text):
    train_source = source_int_text
    train_target = target_int_text
    loss = 0
    # reload the parameters.

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess_new:
        load_path = load_params()
        print(load_path + '.meta')
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess_new, load_path)

        input_data = loaded_graph.get_tensor_by_name('input:0')
        targets = loaded_graph.get_tensor_by_name('targets:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
        lr = loaded_graph.get_tensor_by_name('lr_rate:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

        train_op = loaded_graph.get_collection('optimizer1')
        cost = loaded_graph.get_tensor_by_name('optimization1/cost1/truediv:0')
        for epoch_i in range(epochs):
            for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                    get_batches(train_source, train_target, batch_size,
                                source_vocab_to_int['<PAD>'],
                                target_vocab_to_int['<PAD>'])):
                _, loss = sess_new.run(
                    [train_op, cost],
                    {input_data: source_batch,
                     targets: target_batch,
                     lr: learning_rate,
                     target_sequence_length: targets_lengths,
                     keep_prob: keep_probability})

                print('reloaded Epoch {:>3} Batch {:>4}/{} - Loss: {}'
                      .format(epoch_i, batch_i, len(source_int_text) // batch_size, loss))

        saver = tf.train.Saver()
        saver.save(sess_new, save_path)
        save_params(save_path)
        return loss * 1.0/targets_lengths[0]
