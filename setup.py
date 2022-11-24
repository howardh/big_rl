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
        'scikit-learn',

        # Minigrid
        'minigrid',
        'opencv-python',
        'permutation',
        'scipy',
        'font-roboto==0.0.1',
        'fonts==0.0.3',

        # Atari (Not yet compatible with Gymnasium)
        #'gym[atari,accept-rom-license]',

        # Testing
        'pytest',
    ],
    packages=find_packages()
)
