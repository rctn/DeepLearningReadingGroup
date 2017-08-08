from setuptools import setup, find_packages
from sys import stdin
from os import symlink, mkdir, path

setup(
    name='hippo',
    version = '.1',

    # Choose your license
    license='MIT',
    packages=find_packages(),
    install_requires=['pyyaml', 'h5py'],

)

print('Data directory to link to (blank will create a local directory here):')
data_dir_path = stdin.readline()
if len(data_dir_path) > 1 and not path.exists('data'):
    symlink(path.abspath(data_dir_path[:-1]), 'data')
else:
    if not path.exists('data'):
        mkdir('data')

print('Runs directory to link to (blank will create a local directory here):')
runs_dir_path = stdin.readline()
if len(runs_dir_path) > 1 and not path.exists('runs'):
    symlink(path.abspath(runs_dir_path[:-1]), 'runs')
else:
    if not path.exists('runs'):
        mkdir('runs')
