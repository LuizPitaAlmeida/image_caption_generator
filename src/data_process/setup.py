from setuptools import setup, find_packages

_VERSION = '1.0'

setup(
    name='data_process',
    version=_VERSION,
    packages=find_packages(),
    py_modules=['build_vocab', 'data_loader', 'image_transform'],
    url='https://github.com/LuizPitaAlmeida/image_caption_generator',
)
