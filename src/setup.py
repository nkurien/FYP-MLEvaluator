from setuptools import setup, find_packages
import os
import sys

setup(
    name='Machine Learning Model Evaluator',
    version='1.0.0',
    description='An application to evaluate machine learning models with a graphical user interface.',
    long_description=open('../README.md').read(),
    long_description_content_type='text/markdown',
    author='Nathan Kurien',
    author_email='Nathan.Kurien.2016@live.rhul.ac.uk',
    packages=find_packages(),
    install_requires=[
        'PyQt5',
        'seaborn>=0.12.2',
        'matplotlib>=3.7.3',
        'pandas>=1.1.5',
        'numpy>=1.19.5',
        'scikit-learn',  
        # Add any other dependencies here
    ],
    entry_points={
        'console_scripts': [
            'ml-evaluator = src.main:main',  # Adjust if main.py is not in a src directory
        ],
    },
    package_data={
        # Correct 'ui': ['resources/*'], if 'resources' is a directory inside 'ui'
        'ui': ['images/*'],  # If you store your images here
    },
    tests_require=['pytest'],
    # Include any additional package data
    include_package_data=True,
    # Other metadata
    url='https://gitlab.cim.rhul.ac.uk/zdac117/PROJECT',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.9',  # Specify the minimum required Python version
)

if sys.platform == 'darwin':
    import PyInstaller.__main__
    PyInstaller.__main__.run([
        '--name=Machine Learning Model Evaluator',
        '--onefile',
        '--windowed',
        '--add-data=ui/resources:Resources',
        '--icon=ui/resources/icon.icns',
        'main.py'
    ])