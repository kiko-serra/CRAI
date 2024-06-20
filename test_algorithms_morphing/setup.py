from setuptools import setup, find_packages

setup(
    name='test_algorithms_morphing',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'tqdm'
    ],
    entry_points={
        'console_scripts': [
            'analyse_datasets_accuracies=test_algorithms_morphing.morphing:analyse_datasets_accuracies',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for analyzing and morphing datasets using various machine learning algorithms.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my_ml_package',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)