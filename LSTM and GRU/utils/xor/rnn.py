#!/usr/bin/env python
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import array_ops, nn_ops, init_ops, math_ops
from tensorflow.python.ops import variable_scope as vs


class MyLSTMCell(RNNCell):
    """
    Your own basic LSTMCell implementation that is compatible with TensorFlow. To solve the compatibility issue, this
    class inherits TensorFlow RNNCell class.

    For reference, you can look at the TensorFlow LSTMCell source code. It's located at tensorflow/tensorflow/python/ops/rnn_cell_impl.py.
    If you're using Anaconda, it's located at
    anaconda_install_path/envs/your_virtual_environment_name/site-packages/tensorflow/python/ops/rnn_cell_impl.py

    So this is basically rewriting the TensorFlow LSTMCell, but with your own language.
    Also, you will find Colah's blog about LSTM to be very useful:
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    """

    def __init__(self, num_units, num_proj, forget_bias=1.0, activation=None):
        """
        Initialize a class instance.

        In this function, you need to do the following:

        1. Store the input parameters and calculate other ones that you think necessary.

        2. Initialize some trainable variables which will be used during the calculation.

        :param num_units: The number of units in the LSTM cell.
        :param num_proj: The output dimensionality. For example, if you expect your output of the cell at each time step to be a 10-element vector, then num_proj = 10.
        :param forget_bias: The bias term used in the forget gate. By default we set it to 1.0.
        :param activation: The activation used in the inner states. By default we use tanh.

        There are biases used in other gates, but since TensorFlow doesn't have them, we don't implement them either.
        """
        super(MyLSTMCell, self).__init__(_reuse=None)
        self.num_units = num_units
        self.num_proj = num_proj
        self.forget_bias = forget_bias
        if activation:
            self.activation = activation
        else:
            self.activation = tf.nn.tanh
            
    # The following 2 properties are required when defining a TensorFlow RNNCell.
    @property
    def state_size(self):
        """
        Overrides parent class method. Returns the state size of of the cell.

        state size = num_units + output_size

        :return: An integer.
        
        """
        return self.num_units + self.num_proj

    @property
    def output_size(self):
        """
        Overrides parent class method. Returns the output size of the cell.

        :return: An integer.
        """
        return self.num_proj

    def call(self, inputs, state):
        """
        Run one time step of the cell. That is, given the current inputs and the state from the last time step, calculate the current state and cell output.

        You will notice that TensorFlow LSTMCell has a lot of other features. But we will not try them. Focus on the very basic LSTM functionality.

        Hint 1: If you try to figure out the tensor shapes, use print(a.get_shape()) to see the shape.

        Hint 2: In LSTM there exist both matrix multiplication and element-wise multiplication. Try not to mix them.

        :param inputs: The input at the current time step. The last dimension of it should be 1.
        :param state:  The state value of the cell from the last time step. The state size can be found from function state_size(self).
        :return: A tuple containing (output, new_state). For details check TensorFlow LSTMCell class.
        """
        c = tf.slice(state, [0, 0], [-1, self.num_units])
        h = tf.slice(state, [0, self.num_units], [-1, self.num_proj])
        gates  = tf.contrib.layers.fully_connected(tf.concat([inputs, h], 1),num_outputs=4 * self.num_units, activation_fn = None)                                 
        i, j, f, o = tf.split(gates, 4, 1)

        c = (c * tf.nn.sigmoid(f + self.forget_bias) + tf.nn.sigmoid(i) * tf.nn.tanh(j))
        h =  tf.nn.tanh(c) * tf.nn.sigmoid(o)

        output = tf.contrib.layers.fully_connected(h, num_outputs=2, activation_fn = None)                                 
        new_state = tf.concat([c, output], 1)
        
        return output, new_state