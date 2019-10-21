from setuptools import setup, find_packages

setup(
    name='barebones',
    version='0.01',

    # Package data
    packages=find_packages(),
    include_package_data=True,

    # Insert dependencies list here
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'tensorflow',
        'flask',
        'gunicorn',
        'gevent'
    ],

    entry_points={
        "barebones.training": [
           "train=barebones.train:entry_point"
        ],
        "barebones.hosting": [
           "serve=barebones.server:start_server"
        ]
    }
)
