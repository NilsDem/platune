from setuptools import setup, find_packages

__version__ = "1.0.0"

setup(
    name='platune',
    version=__version__,
    author_email="sarah.nabi@ircam.fr",
    description="PLaTune: Pretained Latents Tuner for controlling pretrained neural audio codecs",
    packages=find_packages(),
)
