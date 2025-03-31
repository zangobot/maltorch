from setuptools import find_packages, setup
setup(
    name='maltorch',
    version='0.1',
    packages=find_packages(
        where="src",
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests",
        ],
    ),
    package_dir={'': 'src'},
    url='',
    license='',
    author='Luca Demetrio, Daniel Gibert, Andrea Ponte, Dmitrijs Trizna',
    author_email='',
    description='',
    install_requires=[
        'torch', 'torchvision', 'scikit-learn', 'secml-torch', 'lightgbm==3.3.5', 'lief',
        'git+https://github.com/zangobot/ember.git'
        , 'nevergrad', 'joblib>=1.3.2'
    ],
)
