from setuptools import setup

setup(
    name='maltorch',
    version='0.1',
    packages=['maltorch', 'maltorch.adv', 'maltorch.adv.evasion', 'maltorch.zoo', 'maltorch.data', 'maltorch.optim',
              'maltorch.utils', 'maltorch.datasets', 'maltorch.trainers', 'maltorch.initializers',
              'maltorch.manipulations', 'maltorch.data_processing', 'maltorch.data_processing.dynamic'],
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
