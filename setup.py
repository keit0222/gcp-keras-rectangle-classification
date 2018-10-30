'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='rectangle_mlp',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='RECTANGLE MLP Keras model on Cloud ML Engine',
      author='Keita Tomochika',
      author_email='gndfr741@gmail.com',
      license='MIT',
      install_requires=[
          'keras',
          'numpy',
          'h5py'],
      zip_safe=False)
