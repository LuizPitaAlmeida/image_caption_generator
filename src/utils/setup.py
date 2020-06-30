from setuptools import setup, find_packages

_VERSION = '1.0'

setup(
    name='utils',
    version=_VERSION,
    packages=find_packages(),
    py_modules=['hardware_stats', 'print_results', 'show_dataset'],
    url='https://github.com/LuizPitaAlmeida/image_caption_generator',
)