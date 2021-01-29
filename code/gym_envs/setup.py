""" Setup file for custom Gym environments (PIP package )"""

from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="widowx_env",
    version="0.0.1",
    author="Pierre Aumjaud",
    author_email="pierre.aumjaud@ucd.ie",
    description="Gym environments for RL reaching tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    download_url='https://pypi.org/project/elastictools/',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    license="MIT",
    keywords=["Reinforcement_Learning", "OpenAI_Gym"],
    install_requires=["gym", "numpy", "pybullet"]
)
