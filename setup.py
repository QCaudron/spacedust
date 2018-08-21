from setuptools import setup

setup(
    name="supernova",
    version="0.1.1",
    description="Blows away all that is unnecessary.",
    url="http://github.com/QCaudron/supernova",
    author="Quentin CAUDRON",
    author_email="quentincaudron@gmail.com",
    license="MIT",
    packages=["supernova"],
    zip_safe=False,
    install_requires=[
        "numpy>=1.13",
        "pandas>=0.22",
        "scikit-learn>=0.19",
        "xgboost>=0.80"
    ]
)
