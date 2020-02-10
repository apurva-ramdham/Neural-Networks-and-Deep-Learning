# this file contains all the necessary data for the task 2 Neural machine translations

import os
import pickle
import copy
import numpy as np
import time
import tensorflow as tf
import pickle


def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()

    return data


CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }

def create_lookup_tables(text):
    # make a list of unique words
    vocab = set(text.split())

    # starts with the special tokens
    vocab_to_int = copy.copy(CODES)

    # the index (v_i) will starts from 4 (the 2nd arg in enumerate() specifies the starting index)
    # since vocab_to_int already contains special tokens
    for v_i, v in enumerate(vocab, len(CODES)):
        vocab_to_int[v] = v_i

    # (2)
    int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab

def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
        source_text, target_text: raw string text to be converted
        source_vocab_to_int, target_vocab_to_int: lookup tables for 1st and 2nd args respectively
    
        return: A tuple of lists (source_id_text, target_id_text) converted
    """
    # empty list of converted sentences
    source_text_id = []
    target_text_id = []
    
    # make a list of sentences (extraction)
    source_sentences = source_text.split("\n")
    target_sentences = target_text.split("\n")
    
    max_source_sentence_length = max([len(sentence.split(" ")) for sentence in source_sentences])
    max_target_sentence_length = max([len(sentence.split(" ")) for sentence in target_sentences])
    
    # iterating through each sentences (# of sentences in source&target is the same)
    for i in range(len(source_sentences)):
        # extract sentences one by one
        source_sentence = source_sentences[i]
        target_sentence = target_sentences[i]
        
        # make a list of tokens/words (extraction) from the chosen sentence
        source_tokens = source_sentence.split(" ")
        target_tokens = target_sentence.split(" ")
        
        # empty list of converted words to index in the chosen sentence
        source_token_id = []
        target_token_id = []
        
        for index, token in enumerate(source_tokens):
            if (token != ""):
                source_token_id.append(source_vocab_to_int[token])
        
        for index, token in enumerate(target_tokens):
            if (token != ""):
                target_token_id.append(target_vocab_to_int[token])
                
        # put <EOS> token at the end of the chosen target sentence
        # this token suggests when to stop creating a sequence
        target_token_id.append(target_vocab_to_int['<EOS>'])
            
        # add each converted sentences in the final list
        source_text_id.append(source_token_id)
        target_text_id.append(target_token_id)
    
    return source_text_id, target_text_id

def preprocess_and_save_data(source_path, target_path, text_to_ids):
    # Preprocess
    
    # load original data (English, French)
    source_text = load_data(source_path)
    target_text = load_data(target_path)

    # to the lower case
    source_text = source_text.lower()
    target_text = target_text.lower()

    # create lookup tables for English and French data
    source_vocab_to_int, source_int_to_vocab = create_lookup_tables(source_text)
    target_vocab_to_int, target_int_to_vocab = create_lookup_tables(target_text)

    # create list of sentences whose words are represented in index
    source_text, target_text = text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int)

    # Save data for later use
    pickle.dump((
        (source_text, target_text),
        (source_vocab_to_int, target_vocab_to_int),
        (source_int_to_vocab, target_int_to_vocab)), open('preproc.pickle', 'wb'))

def load_preprocess():
    with open('preproc.pickle', mode='rb') as in_file:
        return pickle.load(in_file)


def enc_dec_model_inputs():
    """
        This function creates the placeholders for the model inputs
    """
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets') 
    
    target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_length)    
    
    return inputs, targets, target_sequence_length, max_target_len

def hyperparam_inputs():
    lr_rate = tf.placeholder(tf.float32, name='lr_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    return lr_rate, keep_prob

def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for decoder, add the <GO> tag to the start of the senteces
    :return: Preprocessed target data
    """
    # get '<GO>' id
    go_id = target_vocab_to_int['<GO>']
    #drop the last word which is either <PAD> or <EOS>
    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    #add the <GO> tag tp the start of the sentence
    after_concat = tf.concat( [tf.fill([batch_size, 1], go_id), after_slice], 1)
    
    return after_concat

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, 
                   source_vocab_size, 
                   encoding_embedding_size, cell_type):
    """
    creates the encoder part of seq2seq learning
    we will use the paramters:
        rnn_inputs: the input to the encoding layer of the shape (Batch_size,sentence_length)
        rnn_size : the size of the hidden state
        num_layers: number of RNN layers
        keep_prob: the probablity of the dropout layer
        source_vocab_size: size of the voacabulary of the source language
        encoding_embedding_size: size of the embeddings for the encoder
        cell_type: this is the type of cell either 'LSTM' or 'GRU', 

        
        
        returns: 
        rnn_outputs:  this are the hidden states of the RNN unit of the shape [batch_size, max_sentence_length, rnn_size]
        rnn_final_state: this is the state of the last rnn unit

    :return: tuple (rnn_outputs, rnn_final_state)

    Useful functions for implementation:
        tf.nn.rnn_cell.LSTMCell
        tf.nn.rnn_cell.GRUCell
        tf.nn.rnn_cell.DropoutWrapper
        tf.contrib.rnn.MultiRNNCell
        tf.nn.dynamic_rnn
    """
    embed = tf.contrib.layers.embed_sequence(rnn_inputs, 
                                             vocab_size=source_vocab_size, 
                                             embed_dim=encoding_embedding_size)
    ######################################################################################
    # TODO: finish the rnn layer definition, 
    ######################################################################################    
    if cell_type == 'LSTM':
        stacked_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size), keep_prob) for _ in range(num_layers)])
    elif cell_type == 'GRU':
        stacked_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(rnn_size), keep_prob) for _ in range(num_layers)])
        
    return tf.nn.dynamic_rnn(stacked_cells, embed, dtype=tf.float32)


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, 
                         target_sequence_length, max_target_sequence_length, 
                         output_layer, keep_prob):
    """
    Create a training process in decoding layer 
    during training as we have the ground truth output. we use the ground truth output of the last layer as input 
    to the next cell. This is done using TrainingHelper

    args
    encoder_state : the state of the last RNNcell of the encoder
    dec_cell: the RNNcell which will be used in the decoder
    dec_embed_input : the decoder input mapped to the embedding space using tf.nn.embedding_lookup
    target_sequence_length : length of each sentence in target_language
    max_target_sequence_length : maximum length of sentence in target language
    output_layer : the dense layer used to map the rnn_otput to the target_vocaulary_size
    keep_prob: the probablity of the dropout layer
    

    :return: BasicDecoderOutput containing training logits of the shape (batch_size,target_sequence_length,target_vocab_size)
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
    outputs, _, _ =  tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                      impute_finished=True, 
                                                      maximum_iterations=max_target_sequence_length)
    return outputs


def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                        output_layer, batch_size, keep_prob):
    """
    Create a inference process in decoding layer 
    during inference as we do not have the ground truth output. we use the decoder output of the last layer as input 
    to the next cell. This is done using GreedyEmbeddingHelper

    args
    encoder_state : the state of the last RNNcell of the encoder
    dec_cell: the RNNcell which will be used in the decoder
    dec_embeddings : a tensor to map the decoder input to the decoder embeddings
    start_of_sequence_id : id of the <GO> tag
    end_of_sequence_id : id of the <EOS> tag
    max_target_sequence_length : maximum length of sentence in target language
    output_layer : the dense layer used to map the rnn_otput to the target_vocaulary_size
    batch_size : the batch size of the data
    keep_prob: the probablity of the dropout layer
    
    
    :return: BasicDecoderOutput containing inference output of shape (batch_size,target_sentence_length)
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
                   batch_size, keep_prob, decoding_embedding_size,cell_type):
    """
    Create decoding layer of seq2seq learning architecture.
    we will use the paramters:
        dec_input: input to the decoder of the shape (Batch_size,sentence_length)
        encoder_state : the state of the last RNNcell of the encoder
        target_sequence_length : length of each sentence in target_language
        max_target_sequence_length : maximum length of sentence in target language
        rnn_size : the size of the hidden state
        num_layers: number of RNN layers
        target_vocab_to_int : dictionary converting the target vocabulary to unique integers
        target_vocab_size : size of the target voceabulary
        batch_size : batch_size
        keep_prob: the probablity of the dropout layer
        decoding_embedding_size: size of the embeddings for the decoder
        cell_type: this is the type of cell either 'LSTM' or 'GRU', 

        
        
        returns: 
        train_output: output of the type BasicDecoderOutput of the shape (batch_size,target_sequence_length,target_vocab_size)\
                        these contain the logits of the output
        infer_output: output of the inference step of seq2seq learning. This is of the shape (batch_size,target_sequence_length)

    :return: tuple (train_output, infer_output)

    Useful functions for implementation:
        tf.nn.rnn_cell.LSTMCell
        tf.nn.rnn_cell.GRUCell
        tf.nn.embedding_lookup
        tf.contrib.rnn.MultiRNNCell
    """
    ######################################################################################
    # TODO: finish the rnn layer definition,define the following
    # dec_embeddings : a tensor to map the decoder input to the decoder embeddings(used by decoding_layer_infer)
    # dec_embed_input : the decoder input mapped to the embedding space using tf.nn.embedding_lookup
    # cells : stacked RNNcells having num_layers layers
    ######################################################################################
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    if cell_type == 'LSTM':
        cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_layers)])
    elif cell_type == 'GRU':
        cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(rnn_size) for _ in range(num_layers)])  
    ######################################################################################
    # END TODO:
    ######################################################################################
    
    
    #get the output of the decoder during training. This uses the TrainingHelper to get the output
    with tf.variable_scope("decode"):
        output_layer = tf.layers.Dense(target_vocab_size)
        train_output = decoding_layer_train( encoder_state   , 
                                            cells, 
                                            dec_embed_input, 
                                            target_sequence_length, 
                                            max_target_sequence_length, 
                                            output_layer, 
                                            keep_prob)
    #get the output of the decoder during inference. This uses the GreedyEmbeddingHelper to get the output
    with tf.variable_scope("decode", reuse=True):
        infer_output = decoding_layer_infer(encoder_state, 
                                            cells, 
                                            dec_embeddings, 
                                            target_vocab_to_int['<GO>'], 
                                            target_vocab_to_int['<EOS>'], 
                                            max_target_sequence_length, 
                                            output_layer,
                                            batch_size,
                                            keep_prob)

    return (train_output, infer_output)
    


def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int, cell_type):
    """
    Build the Sequence-to-Sequence model
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    enc_outputs, enc_states = encoding_layer(input_data, 
                                             rnn_size, 
                                             num_layers, 
                                             keep_prob, 
                                             source_vocab_size, 
                                             enc_embedding_size,
                                             cell_type)
    
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
                                              dec_embedding_size,
                                              cell_type)
    
    return train_output, infer_output

def my_optimizer(loss,grad_clip, learning_rate):
    '''
    build our optimizer
    Unlike previous worries of gradient vanishing problem,
    for some structures of rnn cells, the calculation of hidden layers' weights 
    may lead to an "exploding gradient" effect where the value keeps growing.
    To mitigate this, we use the gradient clipping trick. Whenever the gradients are updated, 
    they are "clipped" to some reasonable range (like -5 to 5) so they will never get out of this range.
    parameters we will use:
    loss, grad_clip, learning_rate
    :param loss: the final loss calculated by the functions
    :param learning_rate: (float)
    :param grad_clip: constraint of gradient to avoid gradient explosion
    we have to return:
    optimizer for later use
    '''
    # using clipping gradients
    #######################################################
    # TODO: implement your optimizer with gradient clipping
    #######################################################
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss)
    gradients_cap = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var) for grad, var in gradients if grad is not None] 
    
    return optimizer.apply_gradients(gradients_cap)
    #raise NotImplementedError


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


def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))


def save_params(params,cell):
    if cell == "LSTM":
        with open('params_lstm.pickle', 'wb') as out_file:
            pickle.dump(params, out_file)
    elif cell == "GRU":
        with open('params_gru.pickle', 'wb') as out_file:
            pickle.dump(params, out_file)

def load_params(cell):
    if cell == "LSTM":
        with open('params_lstm.pickle', mode='rb') as in_file:
            return pickle.load(in_file)
    elif cell == "GRU":
        with open('params_gru.pickle', mode='rb') as in_file:
            return pickle.load(in_file)

def sentence_to_seq(sentence, vocab_to_int):
    results = []
    for word in sentence.split(" "):
        if word in vocab_to_int:
            results.append(vocab_to_int[word])
        else:
            results.append(vocab_to_int['<UNK>'])
            
    return results