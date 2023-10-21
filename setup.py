from setuptools import setup

setup(
    name='my_word_similarity',
    version='1.0',
    packages=['my_word_similarity'],
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'gensim',
        # Add other dependencies as needed
    ],
)