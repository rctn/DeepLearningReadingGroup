import tensorflow as tf
import yaml
import time
from datetime import datetime
import sys, os
import pprint
import numpy as np
from collections import namedtuple

import hippo.util as util
from hippo.util import Peace
from hippo.networks import Network
import hippo.data as data


def build_network(hparams):
    with tf.name_scope('network_parameters'):
        net = Network(hparams)

    return net

def build_forward_prop_graph(net, n_batch, n_prop, n_reject, summarize, hparams):
    # The network input and teacher signal nodes
    # They reference the same placeholders because we are doing input prediction
    # TODO: update this to the more general context
    # The forward propagation graph
    xs, ys, ys_valid, ys_hat, losses = [], [], [], [], []


    with tf.name_scope('t_0'):
        train_state_store = net.get_new_state_store(n_batch)
        train_state = net.state_from_store(train_state_store)

    last_state = []

    for i in range(n_prop):
        with tf.name_scope('t_' + str(i+1)):
            x_batch = tf.placeholder(tf.float32, shape=[n_batch, *net.input_dim], name = 'x')
            y_batch = tf.placeholder(tf.float32, shape=[n_batch, *net.input_dim], name = 'y')
            y_valid_batch = tf.placeholder(tf.bool, shape=[n_batch], name = 'y_valid')

            xs.append(x_batch)
            ys.append(y_batch)
            ys_valid.append(y_valid_batch)


            train_state, [y_hat_batch] = net.step(train_state, x_batch)
            ys_hat.append(y_hat_batch)

            make_image = (summarize and hparams['data']['type'] == 'video'
                and (i == 1 or i == n_prop - 1))
            if make_image:
                name = 't' + str(i) + '_prediction'
                frame_dim = np.array(hparams['data']['frame_dim'], dtype=int)
                tf.summary.image(name, tf.reshape(y_hat_batch[0], [1, *frame_dim, 1]))

            if i == n_prop - n_reject - 1:
                last_state = train_state

            # We want to ignore first n_bptt_reject outputs on this since
            # their gradient calculations will significantly more biased the
            # later outputs
            if i >= n_reject:
                loss = data.loss(hparams, y_hat_batch, y_batch, y_valid_batch)
                losses.append(loss)

    # This update that allows state to carry across f-props so our network
    # state can carry info across arbitrarily long input histories
    with tf.name_scope('save_and_reset'):
        store_state_op = net.store_state_op(last_state, train_state_store)
        reset_state_op = net.reset_state_op(train_state_store)


    # Our training objective function
    with tf.name_scope('loss_statistics'):
        all_losses = tf.stack(losses, name='all_losses')
        mean_loss = tf.reduce_mean(all_losses, name='mean_loss')

        if summarize:
            tf.summary.histogram('last_y_hat', y_hat_batch)

            std_loss = tf.sqrt(tf.reduce_mean(tf.square(all_losses))
                - tf.square(mean_loss))
            tf.summary.scalar('train_error', mean_loss)
            tf.summary.scalar('train_error_std', std_loss)

    Graph_Components = namedtuple('BPTT_Graph',
        'xs ys ys_valid ys_hat err_func state_store_op reset_state_op')
    graph_components = Graph_Components(xs=xs, ys = ys, ys_valid = ys_valid,
        ys_hat = ys_hat, err_func=mean_loss, state_store_op=store_state_op,
        reset_state_op=reset_state_op)

    return graph_components

def build_train_graph(net, hparams):
    n_batch = hparams['n_batch']
    n_prop = hparams['n_prop']
    n_reject = hparams['n_reject']

    with tf.name_scope('BPTT_Graph'):
        return build_forward_prop_graph(net, n_batch, n_prop, n_reject, True, hparams)

def build_sampling_graph(net, hparams):
    n_batch = 1
    n_prop = 1
    n_reject = 0

    with tf.name_scope('Sampler_Graph'):
        return build_forward_prop_graph(net, n_batch, n_prop, n_reject, False, hparams)

# Given a graph and an error function, add an optimizer to the graph that optimizes
# the error function
def build_optimizer(err_func, hparams):
    opt_params = hparams['optimizer']

    with tf.name_scope('optimizer'):
        t = tf.Variable(0, name= 't', trainable=False) # the step variable

        steps_til_decay = opt_params.get('steps_til_decay', 2000)
        decay_factor = opt_params.get('eta_decay_factor', .5)
        staircase = opt_params.get('staircase', True)
        grad_clip_norm = opt_params.get('grad_clip_norm', 1.0)

        if opt_params['algorithm'] == 'adam':
            eta = tf.train.exponential_decay(opt_params['eta0'], t, steps_til_decay,
                decay_factor, staircase=staircase)
            optimizer = tf.train.AdamOptimizer(learning_rate=eta)
        elif opt_params['algorithm'] == 'sgd':
            eta = tf.train.exponential_decay(opt_params['eta0'], t, steps_til_decay,
                decay_factor, staircase=staircase)
            optimizer = tf.train.GradientDescentOptimizer(eta)
        else:
            raise Exception('I dunno nothing about that optimizer')

        grads, params = zip(*optimizer.compute_gradients(err_func))
        clipped_grads, _ = tf.clip_by_global_norm(grads, grad_clip_norm)
        optimize_op = optimizer.apply_gradients(zip(clipped_grads, params), global_step=t)


    Optimizer = namedtuple('Optimizer', 'optimize_op eta')
    return Optimizer(optimize_op=optimize_op, eta=eta)

def run_training(session, train_components, optimizer, input_producer,
    output_producer, n_train_steps, hparams):
    '''
    Run train the network in graph by optimizing it on the training inputs
        @param graph: the tensorflow graph with bptt graph and optimizer
        @param train_components: components of the bptt_graph (input, err, save, reset...)
        @param optimizer: components of the optimizer graph (optimize_op, eta...)
        @param input_producer: an object to supply input batches
        @param output_producer: an object to supply output batches
        @param save_state_op: op to save state across truncation boundaries
        @param reset_state_op: op to reset state to some initialization
        @param n_train_steps: the number of forward propagation to perform
        @param hparams: bet you couldn't have guessed it's the hyperparameters

        @return: the history the training error

    TODO: start using a simulator and replace the input_generator with a world model ;)
    '''

    n_prop = hparams['n_prop']

    summary_freq = 100
    mean_error = 0.0

    hours_per_checkpoint = hparams.get('hours_per_checkpoint', 2)

    saver = tf.train.Saver(keep_checkpoint_every_n_hours = hours_per_checkpoint,
        save_relative_paths = True, name='graph_saver')

    merged = tf.summary.merge_all()

    if not os.path.exists('/tmp/tf_summaries'):
        os.mkdir('/tmp/tf_summaries')
    writer = tf.summary.FileWriter('/tmp/tf_summaries/', graph=session.graph)

    summ_error = 0.0

    for step in range(n_train_steps):

        # assert 'train_state_reset_rate' in hparams
        # reset_rate = n_prop/hparams['train_state_reset_rate']
        # if np.random.poisson(reset_rate) > 0:
        #     print('State reset!')
        #     session.run([train_components.reset_state_op])

        # Set up input value -> input var mapping
        input_window, _  = input_producer.next_window()
        output_window, is_valid = output_producer.next_window()
        feed_dict = dict()
        for i in range(n_prop):
            feed_dict[train_components.xs[i]] = input_window[i]
            feed_dict[train_components.ys[i]] = output_window[i]
            feed_dict[train_components.ys_valid[i]] = is_valid[i]

        to_compute = [merged, train_components.err_func, optimizer.eta,
            optimizer.optimize_op, train_components.state_store_op]
        summ_str, error_val, eta_val = session.run(to_compute, feed_dict=feed_dict)[:3]
        writer.add_summary(summ_str, step)

        summ_error += error_val

        if step % summary_freq == 0 and step > 0:
            mean_error = summ_error/summary_freq

            train_error_hist.append((step, mean_error))

            print('Average error at step', step, ':', mean_error, 'learning rate:', eta_val)
            summ_error = 0.0

            # TODO: validation error is for punks


    # save the model, params, and stats to the run dir
    model_file = os.path.join(hparams['run_path'], 'model.ckpt')

    saver.save(session, model_file)

    return util.nested_list_to_array(train_error_hist)

def sample(session, sampler_components, seed, n_sample_steps, hparams):
    n_seed = seed.shape[0]

    input_producer = data.gen_one_step_producer(seed, hparams)
    samples = []

    y_hat = sampler_components.ys_hat[0]
    if hparams['data']['type'] == 'video':
        dist = y_hat # we currently just predict the mean so there's no good way of sampling
    elif hparams['data']['type'] == 'text':
        dist = tf.nn.softmax(y_hat)
    else:
        raise data.DataTypeException()

    to_compute = [dist, sampler_components.state_store_op]

    for step in range(n_seed):
        input_window, _ = input_producer.next_window()
        # import ipdb; ipdb.set_trace()

        feed_dict = {sampler_components.xs[0]: input_window[0]}
        dist, _ = session.run(to_compute, feed_dict = feed_dict)

        prediction = data.sample_distribution(dist, hparams)


    for step in range(n_sample_steps):
        prediction = data.sample_distribution(dist, hparams)
        samples.append(prediction)

        if step == n_sample_steps - 1:
            break # We already have sampled n_sample_steps times

        feed_dict = {sampler_components.xs[0]: prediction}
        dist, _ = session.run(to_compute, feed_dict = feed_dict)

    return np.stack(samples)


if __name__ == '__main__':

    np.random.seed(0)
    tf.set_random_seed(0)

    t_run_start = time.time()

    if len(sys.argv) == 2:
        main_param_filename = sys.argv[1]
    elif len(sys.argv) > 2:
        raise Exception('I only take one argument you dingus!')
    else:
        raise Exception('Specify a parameter file to run on you dingus!')

    hparams = util.load_params(main_param_filename)

    print('-------------')
    print('Running on the following params:')
    pprint.pprint(hparams)
    print('-------------')

    if 'start_date' not in hparams:
        start_date = datetime.fromtimestamp(t_run_start)
        date_str = start_date.isoformat()
        hparams['start_date'] = date_str
    else:
        date_str = hparams['start_date']

    assert 'run_name' in hparams, 'You must specify a run name in the params'
    run_dir  = hparams.get('run_dir', 'runs')
    run_id = hparams['run_name'] + '-' + date_str
    hparams['run_id'] = run_id
    run_path = os.path.join(run_dir, run_id)
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    hparams['run_path'] = run_path

    train_data, valid_data, test_data = data.gen_datasets(hparams)

    input_producer, output_producer = data.gen_prediction_data_producers(train_data, hparams)

    # Set the input and output layer dimensions to the the data dimension
    hparams['layers'][0]['dim'] = hparams['data']['dim']
    hparams['layers'][-1]['dim'] = hparams['data']['dim']

    print('Building the graph...')
    graph = tf.Graph()
    with graph.as_default():
        net = build_network(hparams)
        train_components = build_train_graph(net, hparams)
        optimizer = build_optimizer(train_components.err_func, hparams)
        sampler_components = build_sampling_graph(net, hparams)


    with tf.Session(graph=graph) as session:
        init_op = tf.global_variables_initializer()
        session.run(init_op)

        print('Running training...')
        train_error_hist = run_training(session, train_components, optimizer,
            input_producer, output_producer, hparams['n_train_steps'], hparams)

        print('Do some sampling...')
        if hparams['data']['type'] == 'video':
            sample_seed = train_data[0]
        elif hparams['data']['type'] == 'text':
            sample_seed = train_data[:100]
        n_samples = hparams['n_samples']
        samples = sample(session, sampler_components, sample_seed, n_samples, hparams)
        data.save(samples, hparams)

    print('Writing summary data...')

    stats = {'train_error_hist': train_error_hist}
    stats_file = os.path.join(run_path, 'stats.hdf5')
    hippo.util.write_dict_to_hdf5(stats_file, stats)

    params_file = os.path.join(run_path, 'hparams.yaml')
    with open(params_file, 'w') as f:
        yaml.dump(hparams, f)

    # Don't forget to include a proper signoff!
    p = Peace()
    p.out()
