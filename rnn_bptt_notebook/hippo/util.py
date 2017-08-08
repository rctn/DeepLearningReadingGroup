"""
This module probably shouldn't exists
"""
import h5py
import os
import yaml
import numpy as np
import tensorflow as tf
from IPython.display import display, HTML


class Peace(object):
    def __init__(self):
        pass

    def out(self):
        print('Peace out!')



def write_dict_to_hdf5(filename, dictionary):
    with h5py.File(filename, 'w') as hdf5_file:
        for key in dictionary:
            hdf5_file.create_dataset(key, data=dictionary[key])


def load_params(param_filename):
    main_param_path = os.path.join('params', param_filename)

    params = {}
    with open(main_param_path, 'r') as f:
        main_params = yaml.load(f)
    if 'use_params' in main_params:
        for sub_param_filename in main_params['use_params']:
            sub_params = load_params(sub_param_filename)

            recursive_update(params, sub_params)

    recursive_update(params, main_params)

    return params

def recursive_update(old_hash, new_hash):
    for key in new_hash:
        if key in old_hash and type(old_hash[key]) is dict:
            recursive_update(old_hash[key], new_hash[key])
        else:
            old_hash[key] = new_hash[key]

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

def nested_list_to_array(nested_list):
    arr = np.zeros((len(nested_list), len(nested_list[0])))
    for i, vals in enumerate(zip(*nested_list)):
        arr[:, i] = np.array(vals)
    return arr
