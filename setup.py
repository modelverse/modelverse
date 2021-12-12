import os

from setuptools import find_packages, setup


def get_version() -> str:
    version_filepath = os.path.join(os.path.dirname(__file__), "python-package", "modelverse", "version.py")
    with open(version_filepath) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split()[-1][1:-1]
    assert False


def get_long_description():
    readme_filepath = os.path.join(os.path.dirname(__file__), "README.md")
    with open(readme_filepath) as f:
        return f.read()


def get_install_requires():
    requirements = [
    ]
    return requirements


setup(
    name="modelverse",
    version=get_version(),
    author="Aakansh Gupta",
    author_email="modelverse.n77py@simplelogin.co",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    packages=find_packages(where="./python-package"),
    package_dir={'': 'python-package'},
    python_requires=">=3.6",
    install_requires=get_install_requires(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
    ],
)
