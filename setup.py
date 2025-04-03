import pathlib

from setuptools import find_packages, setup

here = pathlib.Path.cwd()
readme_path = here / "README.md"
version_path = here / "src" / "maltorch" / "VERSION"

CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: Implementation :: PyPy
Topic :: Software Development
Topic :: Scientific/Engineering
"""

# Get the long description from the README file
with readme_path.open() as f:
    long_description = f.read()

# Get the version from VERSION file
with version_path.open() as f:
    version = f.read()

setup(
    name='maltorch',
    version=version,
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
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
    data_files=[("src/maltorch/VERSION", ["src/maltorch/VERSION"])],
    url='',
    license='MIT',
    author='Luca Demetrio, Daniel Gibert, Andrea Ponte, Maura Pintor',
    author_email='luca.demetrio@unige.it',
    description='Pytorch-based library for creating Adversarial EXEmples against Windows Malware detectors.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'torch',
        'torchvision',
        'secml-torch',
        'lightgbm',
        'lief',
        'nevergrad',
        'joblib>=1.3.2'
    ],
)
