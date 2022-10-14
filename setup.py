from setuptools import setup, find_packages

setup(name='big_rl',
    version='0.0.1',
    install_requires=[
        'gymnasium',
        'torch>=1.11.0',
        'torchtyping',
        'matplotlib',
        'tqdm',
        'wandb',

        # Minigrid
        'minigrid',
        'opencv-python',
        'permutation',
        'scipy',

        # Atari
        #'gym[atari,accept-rom-license]',

        # Testing
        'pytest',
    ],
    packages=find_packages()
)
