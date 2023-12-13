from setuptools import setup, find_packages
#from torch.utils import cpp_extension

setup(name='big_rl',
    version='0.0.1',
    install_requires=[
        'gymnasium>=0.29.0',
        'torch>=2.0.0',
        'torchtyping',
        'jaxtyping',
        'matplotlib',
        'tqdm',
        'wandb',
        'scikit-learn',
        'tabulate>=0.9.0',
        'tensordict>=0.1.0',
        'pydantic>=2.1.1',
        'simple_slurm',

        # Minigrid
        'minigrid',
        'opencv-python',
        'permutation',
        'scipy',
        'font-roboto==0.0.1',
        'fonts==0.0.3',

        # Mujoco
        'gymnasium[mujoco]',
        'beautifulsoup4', # Needed for manipulating MJCF files
        'lxml',

        # Atari
        'gymnasium[atari,accept-rom-license]',

        # Dev stuff
        'pytest',
        'pytest-timeout',
        'flake8',
    ],
    packages=find_packages(),
    #package_dir={'': '.'},
    #package_data={
    #    #'big_rl': ['big_rl/mujoco/envs/assets/*.xml']
    #    'big_rl.mujoco.envs.assets': ['*.xml']
    #    #'big_rl': ['*.xml']
    #},
    #ext_modules=[
    #    cpp_extension.CppExtension(
    #        name='big_rl_cpp',
    #        sources=['big_rl/model/cpp/batch_linear.cpp'],
    #        extra_compile_args=[
    #            # Use lines below if we need to specify include paths. Replace the ??? with the correct path.
    #            #'-I???/torch/include',
    #            #'-I???/torch/include/torch/csrc/api/include',
    #        ],
    #    )
    #],
    #cmdclass={'build_ext': cpp_extension.BuildExtension},
)
