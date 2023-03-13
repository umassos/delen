from setuptools import setup
from importlib.machinery import SourceFileLoader

version = SourceFileLoader("delen.version", "delen/version.py").load_module()

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='delen',
    version=version.version,
    packages=['delen', 'delen.utils', 'delen.models'],
    url='https://github.com/umassos/delen',
    license='MIT License',
    author='Qianlin Liang',
    author_email='qliang@cs.umass.edu',
    description='Dělen: Enabling Flexible and Adaptive Model-serving',
    long_description=long_description,
    long_description_content_type="text/markdown"
)


