from setuptools import setup
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
setup(
    name='ImageDataAugmenter',
    version='0.0.10',
    description='Augment Image Data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/adityamangal1998/Image-Data-Augmenter.git',
    author='Aditya Mangal',
    author_email='adityamangal98@gmail.com',
    license="MIT",
    packages=['ImageDataAugmenter'],
    install_requires=['opencv-python>=4.5',
                      'numpy>=1.17',
                      'future>=0.17',
                      'glob2>=0.7'
                      ],

    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
)