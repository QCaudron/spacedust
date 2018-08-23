from setuptools import setup
import spacedust

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name="spacedust",
    version=spacedust.__version__,
    description="Blows away all that is unnecessary.",
    long_description=readme(),
    url="http://github.com/QCaudron/spacedust",
    author="Quentin CAUDRON",
    author_email="quentincaudron@gmail.com",
    license="MIT",
    packages=["spacedust"],
    zip_safe=False,
    install_requires=[
        "numpy>=1.13",
        "pandas>=0.22",
        "scikit-learn>=0.19",
        "xgboost>=0.80"
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 4 - Beta"
    ]
)
