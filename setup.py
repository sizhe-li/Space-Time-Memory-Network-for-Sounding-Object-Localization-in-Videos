from setuptools import setup

install_requires = ['numpy', 'torch', 'torchvision', 'librosa', 'natsort', 'torchaudio', 'opencv-python', 'tqdm',
                    'tensorboard', 'yacs', 'youtube_dl', 'einops', 'moviepy', 'h5py', 'pandas']

setup(name="sol",
      version="0.0.1",
      install_requires=install_requires
      )
