from setuptools import setup
import sys

if sys.version_info < (3, 6):
    sys.exit('Sorry, only Python >= 3.6 is supported')

setup(
    python_requires='>=3.6, <4',
    install_requires=[
        'Pillow>=4.1.1',
        'networkx>=2.0',
        'numpy>=1.12.1',
        'opencv-python>=4.0.0.21',
        'pandas>=0.21.1',
        'pathlib2;python_version<"3.5"',
        'scikit-image>=0.13.1',
        'scikit-learn>=0.18',
        'scipy>=1.0.0',
        'tqdm>=4.28.1'
    ])
