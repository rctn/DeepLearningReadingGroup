import copy

import tensorflow as tf
import hippo.layers as layers


class Network(object):
    def __init__(self, hparams):

        self.layers = []
        self.capture_output = []

        self.input_dim = []

        # create one or more rnn layers
        layer_input_dim = []
        for i, layer_hparams in enumerate(hparams['layers']):
            layer_dim = layer_hparams['dim']


            if i == 0:
                self.input_dim = layer_dim
                layer_input_dim = layer_dim
                continue

            layer_name = ''
            if 'name' in layer_hparams:
                layer_name = layer_hparams['name']
            else:
                layer_name = layer_hparams['type'] + str(i)

            if 'type' in layer_hparams:
                layer_type = layer_hparams['type']
            elif 'default_layer_type' in hparams:
                layer_type = hparams['default_layer_type']
            else:
                raise ValueError('but but I dont know what type layer you want!')



            if 'capture_output' in layer_hparams:
                self.capture_output.append(layer_hparams['capture_output'])
            elif i == len(hparams['layers' ]) - 1: # capture the last layer output
                self.capture_output.append(True)
            else:
                self.capture_output.append(False)


            with tf.name_scope(layer_name):

                if layer_type == 'SRNN':
                    layer = layers.SRNN_Layer(layer_name, layer_input_dim, layer_dim, hparams)
                elif layer_type == 'GRU':
                    layer = layers.GRU_Layer(layer_name, layer_input_dim, layer_dim, hparams)
                elif layer_type == 'Linear':
                    layer = layers.Linear_Layer(layer_name, layer_input_dim, layer_dim, hparams)
                else:
                    raise ValueError('das not a layer type honey!')
            self.layers.append(layer)


            layer_input_dim = layer_dim


    def step(self, state, x):
        layer_input = x

        outputs = []
        new_state = []

        with tf.name_scope('step'):
            for i, layer in enumerate(self.layers):

                layer_state = state[i]

                if not layer_state == []:
                    layer_output = layer.step(layer_state, layer_input)
                    # NOTE: this assumes the new state is layer output
                    layer_state = layer_output
                else:
                    layer_output = layer.step(layer_input)

                if self.capture_output[i]:
                    outputs.append(layer_output)

                layer_input = layer_output

                new_state.append(layer_state)

        return new_state, outputs


    def get_new_state_store(self, n_state):
        state_store = []
        for layer in self.layers:
            layer_store = []

            if layer.has_state:
                layer_store = tf.Variable(tf.zeros([n_state, *layer.dim]),
                    trainable=False, name='h_'+ layer.name)
            state_store.append(layer_store)

        return state_store

    # Basically just makes a new set of pointers to the state store
    def state_from_store(self, state_store):
        # DANGER
        # This assumes the layer_store is something that can just be shallow copied
        return [copy.copy(layer_store) for layer_store in state_store]

    # Records current state of the network in storage
    def store_state_op(self, state, state_store):
        store_ops = []
        for i in range(len(self.layers)):
            layer_state = state[i]
            if layer_state == []:
                continue
            else:
                store_op = state_store[i].assign(layer_state)
                store_ops.append(store_op)

        return tf.group(*store_ops)


    # Resets the network state to zero
    def reset_state_op(self, state):
        reset_ops = []
        for i in range(len(self.layers)):
            layer_state = state[i]

            if layer_state == []:
                continue
            else:
                reset_op = layer_state.assign(tf.zeros(layer_state.get_shape()))
                reset_ops.append(reset_op)

        return tf.group(*reset_ops)
