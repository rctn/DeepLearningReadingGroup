import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
import os
#plt.ion()


class Char_Producer(object):
    def __init__(self, text, alphabet, n_batch, n_prop, n_reject):
        ''' Takes the input text as list of indices into the alphabet'''
        self.n_batch = n_batch
        self.n_prop = n_prop
        self.n_reject = n_reject

        self.text = text
        self.alphabet = alphabet

        n_text = len(text)
        self.text_ptrs = np.arange(0, n_text, np.floor(n_text/n_batch), dtype = np.int)

    def next_window(self):
        window = np.zeros((self.n_prop, self.n_batch, self.alphabet.size))
        is_valid = np.ones((self.n_prop, self.n_batch), dtype=bool)

        for batch_idx in range(self.n_batch):
            text_ptr = self.text_ptrs[batch_idx]

            for prop_idx in range(self.n_prop):
                text_idx = (text_ptr + prop_idx) % len(self.text)
                char_id = self.text[text_idx]

                window[prop_idx, batch_idx, char_id] = 1.0

            self.text_ptrs[batch_idx] = (text_ptr + self.n_prop - self.n_reject) % len(self.text)

        return window, is_valid

class Frame_Producer(object):
    def __init__(self, videos, n_batch, n_prop, n_reject):

        self.n_batch = n_batch
        self.n_prop = n_prop
        self.n_reject = n_reject

        self.frame_shape = videos[0].shape[1:]

        self.video_ptrs = np.arange(0, self.n_batch, dtype=np.int32)
        self.frame_ptrs = np.zeros(self.n_batch, dtype=np.int32)

        self.videos = videos

    def next_window(self, flatten_frames = True):
        is_valid = np.ones((self.n_prop, self.n_batch), dtype=bool)
        if flatten_frames:
            window = np.zeros((self.n_prop, self.n_batch, np.prod(self.frame_shape)))
        else:
            window = np.zeros((self.n_prop, self.n_batch, *self.frame_shape))

        for batch_idx in range(self.n_batch):
            video_idx = self.video_ptrs[batch_idx]
            frame_idx = self.frame_ptrs[batch_idx]

            video = self.videos[video_idx]

            for prop_idx in range(self.n_prop):
                if frame_idx + prop_idx < len(video):
                    frame = video[frame_idx + prop_idx, :, :]
                    if flatten_frames:
                        window[prop_idx, batch_idx, :] = frame.flat
                    else:
                        window[prop_idx, batch_idx, :, :] = frame
                else:
                    is_valid[prop_idx, batch_idx] = False

            if frame_idx + self.n_prop >= len(video):
                self.video_ptrs[batch_idx] = (video_idx + self.n_batch) % len(self.videos)
                self.frame_ptrs[batch_idx] = 0
            else:
                self.frame_ptrs[batch_idx] += self.n_prop - self.n_reject

        return window, is_valid

class DataTypeException(Exception):
    def __init__(self):
        Exception.__init__('Yo! WTF is the that data type genius?')


def gen_datasets(hparams):
    '''
    Generate or load a the desired data set and split it into train/validate/test
        @param hparams: the hyperparameters
        @return: train_data, validate_data, test_data
    '''

    data_split = hparams.get('data_split', [.8, .1, .1])

    if hparams['data']['type'] == 'video':
        videos = gen_bouncing_block_dataset(hparams)
        hparams['data']['dim'] = [np.prod(hparams['data']['frame_dim'])]
        return split_data(videos, data_split)
    elif hparams['data']['type'] == 'text':
        alphabet, text = load_text_dataset(hparams)
        hparams['data']['alphabet'] = alphabet
        hparams['data']['dim'] = [alphabet.size]
        return split_data(text, data_split)
    else:
        raise DataTypeException()

def save(data, hparams, name='sample'):
    if hparams['data']['type'] == 'video':
        write_video(data, hparams, name)
    elif hparams['data']['type'] == 'text':
        write_text(data, hparams, name)
    else:
        raise DataTypeException()

def write_video(data, hparams, name):

    Writer = animation.writers['ffmpeg']
    metadata = dict(title=name, artist='A very clever neural net')
    writer = Writer(fps=30, metadata=metadata)

    fig = plt.figure()

    first_frame = np.reshape(data[0,:,:], hparams['data']['frame_dim'])
    axis_image = plt.imshow(first_frame, cmap="Greys_r", interpolation="none",
        animated=True)
    axis_image.set_clim(vmin=0.0, vmax=1.0)

    path = os.path.join(hparams['run_path'], name + '.mp4')

    with writer.saving(fig, path, len(data)):
        for vec in data:
            frame = np.reshape(vec, hparams['data']['frame_dim'])
            axis_image.set_array(frame)
            writer.grab_frame()

    # axis_image = plt.imshow(data[0,:,:], cmap="Greys_r", interpolation="none", animated=True)
    # axis_image.set_clim(vmin=0.0, vmax=1.0)


    # def animate(i):
    #     frame = np.reshape(data[i,:,:], hparams['data']['frame_dim'])
    #     axis_image.set_array(frame)
    #     return axis_image,

    # fig = plt.figure()

    # ani = animation.FuncAnimation(fig, animate, frames=len(data))
    # ani.save('/tmp/animation.gif', writer='imagemagick', fps=30)

def write_text(data, hparams, name):
    text_chars = [onehot_to_char(vec, hparams['data']['alphabet']) for vec in data]
    text_string = ''.join(text_chars)

    path = os.path.join(hparams['run_path'], name + '.txt')

    with open(path, 'w') as f:
        f.write(text_string)

def gen_prediction_data_producers(train_data, hparams):
    if hparams['data']['type'] == 'video':
        return gen_video_prediction_data_producers(train_data, hparams)
    elif hparams['data']['type'] == 'text':
        return gen_text_prediction_data_producers(train_data, hparams['data']['alphabet'], hparams)
    else:
        raise DataTypeException()

def gen_one_step_producer(data, hparams):
    if hparams['data']['type'] == 'video':
        return Frame_Producer([data], 1, 1, 0)
    elif hparams['data']['type'] == 'text':
        return Char_Producer(data, hparams['data']['alphabet'], 1, 1, 0)
    else:
        raise DataTypeException()

def gen_text_prediction_data_producers(train_text, alphabet, hparams):
    prediction_delay = np.int(hparams['prediction_delay'])

    output_text = np.concatenate((train_text[prediction_delay:], train_text[:prediction_delay]))

    input_producer = Char_Producer(train_text, alphabet, hparams['n_batch'],
            hparams['n_prop'], hparams['n_reject'])
    output_producer = Char_Producer(output_text, alphabet, hparams['n_batch'],
            hparams['n_prop'], hparams['n_reject'])

    return input_producer, output_producer

def gen_video_prediction_data_producers(train_videos, hparams):

    prediction_delay = np.int(hparams['prediction_delay'])

    input_videos = []
    output_videos = []
    for i in range(len(train_videos)):
        input_videos.append(train_videos[i][:-prediction_delay])
        output_videos.append(train_videos[i][prediction_delay:])


    input_producer = Frame_Producer(input_videos, hparams['n_batch'],
            hparams['n_prop'], hparams['n_reject'])
    output_producer = Frame_Producer(output_videos, hparams['n_batch'],
            hparams['n_prop'], hparams['n_reject'])

    return input_producer, output_producer

def loss(hparams, *args):
    if hparams['data']['type'] == 'video':
        y, y_hat, is_valid = args
        return l2_loss(y, y_hat, is_valid)
    elif hparams['data']['type'] == 'text':
        logits, p_true, is_valid = args
        #TODO: use the valid bits rather than just ignoring they exist
        return tf.losses.softmax_cross_entropy(p_true, logits)
    else:
        raise DataTypeException()

def sample_distribution(dist, hparams):
    if hparams['data']['type'] == 'video':
        return dist # we currently just predict the mean so there's no good way of sampling
    elif hparams['data']['type'] == 'text':
        return sample_softmax(dist, hparams)
    else:
        raise DataTypeException()

def sample_softmax(dist, hparams):
    bias = hparams['data'].get('sampling_bias', 0.0)

    logits = np.log(dist)
    biased_dist = np.exp((1 + bias)*logits)
    biased_dist = biased_dist/np.sum(biased_dist)

    threshold = np.random.rand()

    # Makes a sample with a 1 in the first spot where it's greater than the
    # threshold and zeros everywhere else
    above = np.cumsum(dist) >= threshold
    sample = above
    sample[1:] = np.logical_and(above[1:], np.logical_not(above[:-1]))

    return np.stack([np.float32(sample)])

def softmax_cross_entropy(y_pred, p_true, is_valid):
    p_pred = tf.nn.softmax(y_pred)
    return cross_entropy(p_pred, p_true, is_valid)

def cross_entropy(p_pred, p_true, is_valid):
    eps = 1e-13
    p_safe_pred = tf.maximum(p_pred, eps)

    return -1.0*tf.reduce_sum(tf.boolean_mask(p_true*tf.log(p_safe_pred), is_valid),
        1, keep_dims=True)

def l2_loss(y, y_hat, is_valid):
    return tf.nn.l2_loss(tf.boolean_mask(y - y_hat, is_valid))

def get_split_pts(n_data, data_split):
    eps = 1e-6
    assert abs(sum(data_split) - 1.0) < eps

    train_valid_pt  = np.int(np.floor(n_data*data_split[0]))
    valid_test_pt = np.int(np.floor(n_data*(data_split[0] + data_split[1])))

    return train_valid_pt, valid_test_pt

def split_data(data, data_split):
    split_pts = get_split_pts(len(data), data_split)

    train_data = data[:split_pts[0]]
    validate_data = data[split_pts[0]:split_pts[1]]
    test_data = data[split_pts[1]:]

    return train_data, test_data, validate_data

def load_text_dataset(hparams):
    dparams = hparams['data']

    corpus_path = dparams['corpus_path']
    to_lower = dparams['to_lower']

    with open(corpus_path, 'r') as corpus_file:
        text = corpus_file.read()
        if to_lower:
            text = text.lower()

    alphabet, text = string_to_alphabet_indices(text)

    return alphabet, text

def gen_bouncing_block_dataset(hparams):
    dparams = hparams['data']
    n_frames = dparams['n_frames']
    frame_dim = np.array(dparams['frame_dim'], dtype=np.int)
    block_size = np.array(dparams['block_size'], dtype=np.int)
    n_video = dparams['n_video']
    v_range = np.array(dparams['v_range'], dtype=np.int)

    videos = []

    x0 = np.zeros(2, dtype=np.int)
    v = np.zeros(2, dtype=np.int)
    for _ in range(n_video):
        for i in range(2):
            x0[i] = np.random.randint(0, frame_dim[i] - block_size[i])
            v[i] = np.random.randint(v_range[0], v_range[1]+1)

        videos.append(gen_bouncing_block_video(n_frames, frame_dim, block_size,
            x0, v))

    return videos

def gen_bouncing_block_video(n_frames, frame_dim, block_size, x0, v):
    video = np.zeros([n_frames, *frame_dim])

    x = x0

    for frame_idx in range(n_frames):
        video[frame_idx, x[0]:x[0]+block_size[0], x[1]:x[1]+block_size[1]] = 1.0

        x += v

        for i in range(2):
            if x[i] + block_size[i] > frame_dim[i]:
                v[i] = -v[i]
                x[i] += 2*(frame_dim[i] - x[i] - block_size[i])
            if x[i] < 0:
                v[i] = -v[i]
                x[i] = -x[i]
    return video

def char_to_onehot(char, alphabet):
    alpha_id = np.where(alphabet == char)[0][0]
    return id_to_onehot(alpha_id, alphabet)

def id_to_onehot(alpha_id, alphabet):
    input_val = np.zeros([1, len(alphabet)], dtype=np.float32)
    input_val[0, alpha_id] = 1
    return input_val

def onehot_to_char(vec, alphabet):
    return alphabet[np.argmax(vec)]

def string_to_alphabet_indices(string):
    '''Finds the alphabet used in string and returns it along with an integer
    array that re-enodes each character in the string to its integer order in
    the alphabet'''
    counter = Counter(string)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    alphabet, _ = list(zip(*count_pairs))
    alphabet = np.array(alphabet)
    alpha_ids = dict(zip(alphabet, range(len(alphabet))))
    indices = np.array([alpha_ids.get(char) for char in string])

    return alphabet, indices

def main():
    n_frames = 100
    frame_dim = [100, 100]
    block_size = [10, 5]
    x0 = np.array([0, 1])
    v = np.array([2, 3])

    video = gen_bouncing_block_video(n_frames, frame_dim, block_size, x0, v)

    fig = plt.figure()

    axis_image = plt.imshow(video[0,:,:], cmap="Greys_r", interpolation="none", animated=True)
    axis_image.set_clim(vmin=0.0, vmax=1.0)

    fig.canvas.draw()

    def update(i):
        axis_image.set_array(video[i,:,:])
        return axis_image,

    ani = animation.FuncAnimation(fig, update, frames=range(video.shape[0]), interval=50, blit=True)

    plt.show()

    hparams = dict()
    hparams['data'] = dict()
    hparams['data']['frame_dim'] = frame_dim
    hparams['run_path'] = '/tmp'

    write_video(video, hparams, 'test')

    return

if __name__ == '__main__':
    main()
