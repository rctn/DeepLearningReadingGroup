import tensorflow as tf

def init_weights(n_input, n_unit, hparams):

    # Every ones knows this is how you do it of course ;)
    return tf.truncated_normal([n_input, n_unit], 0.0,
        tf.sqrt(2.0/tf.cast(n_input + n_unit, tf.float32)))


class Layer(object):
    def __init__(self, name, has_state, input_dim, layer_dim):
        self.name = name
        self.has_state = has_state
        self.input_dim = input_dim
        self.dim = layer_dim

class Linear_Layer(Layer):
    def __init__(self, name, input_dim, layer_dim, hparams):
        Layer.__init__(self, name, False, input_dim, layer_dim)

        n_input = input_dim[0]
        n_unit = layer_dim[0]

        self.W = tf.Variable(init_weights(n_input, n_unit, hparams), name='W')
        self.b = tf.Variable(tf.zeros([1, n_unit]), name='b')

    def step(self, x):
        return tf.matmul(x, self.W) + self.b

class Softmax_Linear_Layer(Layer):
    def __init__(self, name, input_dim, layer_dim, hparams):
        Layer.__init__(self, name, False, input_dim, layer_dim)

        self.linear = Linear_Layer(name, input_dim, layer_dim, hparams)

    def step(self, x):
        return tf.nn.softmax(self.linear.step(x))


class SRNN_Layer(Layer):
    def __init__(self, name, input_dim, layer_dim, hparams):
        Layer.__init__(self, name, True, input_dim, layer_dim)

        n_input = input_dim[0]
        n_unit = layer_dim[0]
        self.W = tf.Variable(init_weights(n_input, n_unit, hparams), name='W')
        self.R = tf.Variable(init_weights(n_unit, n_unit, hparams), name='R')
        self.b = tf.Variable(tf.zeros([1, n_unit]), name='b')

    def step(self, state, x):
        u = tf.matmul(x, self.W) + tf.matmul(state, self.R) + self.b
        return tf.tanh(u)

class GRU_Layer(Layer):
    def __init__(self, name, input_dim, layer_dim, hparams):
        Layer.__init__(self, name, True, input_dim, layer_dim)

        n_input = input_dim[0]
        n_unit = layer_dim[0]

        # The new state weights
        self.W = tf.Variable(init_weights(n_input, n_unit, hparams), name='W')
        self.R = tf.Variable(init_weights(n_unit,n_unit, hparams), name='R')
        self.b = tf.Variable(tf.zeros([1, n_unit]), name='b')

        # The update gate weights
        self.Wu = tf.Variable(init_weights(n_input, n_unit, hparams), name='W')
        self.Ru = tf.Variable(init_weights(n_unit,n_unit, hparams), name='R')
        self.bu = tf.Variable(tf.zeros([1, n_unit]), name='b')

        # The reset gate weights
        self.Wr = tf.Variable(init_weights(n_input, n_unit, hparams), name='W')
        self.Rr = tf.Variable(init_weights(n_unit,n_unit, hparams), name='R')
        self.br = tf.Variable(tf.zeros([1, n_unit]), name='b')


    def step(self, state, x):
        reset_gate = tf.sigmoid(tf.matmul(x, self.Wr) +
            tf.matmul(state, self.Rr) + self.br)
        new_state = tf.sigmoid(tf.matmul(x, self.W) +
            reset_gate*tf.matmul(state, self.R) + self.b)
        update_gate = tf.sigmoid(tf.matmul(x, self.Wr) +
            tf.matmul(state, self.Rr) + self.br)

        return update_gate*new_state + (1 - update_gate)*new_state
