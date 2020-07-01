from setuptools import setup, find_packages

_VERSION = '1.0'

setup(
    name='imgcap_models',
    version=_VERSION,
    packages=find_packages(),
    py_modules=['bert', 'decoder', 'encoder', 'lightning_model',
                'softattention'],
    url='https://github.com/LuizPitaAlmeida/image_caption_generator',
)
