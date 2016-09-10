from setuptools import setup
import os

try:
   import pypandoc
   long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
   long_description = open('README.md').read()

setup(name='multipy',
      version='0.0.1',
      description='Augment data based on its semantic type',
      long_description=long_description,
      url='https://github.com/popily/multipy',
      download_url ='https://github.com/popily/multipy/tarball/0.0.1',
      author='Jonathon Morgan',
      author_email='jonathon@popily.com',
      license='MIT',
      packages=['penny'],
      test_suite='tests',
      install_requires=['python-dateutil','address','phonenumbers'],
      package_data = {
            'penny': ['data/*.csv'],
      },
      zip_safe=False)