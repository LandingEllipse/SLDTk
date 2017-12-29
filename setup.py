from os import path
from codecs import open
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="sldtk",
    version="0.1.0",
    description="The Solar Limb Darkening Toolkit",
    long_description=long_description,
    url="https://github.com/Legendin/SLDTk",
    author="Ariel Ladegaard",
    author_email="legendin@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="sun solar limb darkening toolkit correction image processing "
             "analysis modelling",
    packages=find_packages(),
    install_requires=[
        "numpy>=1",
        "opencv-python>=3",
        "matplotlib>=2",
    ],
    entry_points={
        'console_scripts': [
            'sldtk=sldtk:cli',
        ],
    },
    python_requires='>=3',
)
