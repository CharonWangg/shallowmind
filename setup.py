from setuptools import setup, find_packages

setup(
    name="shallowmind",
    version="0.0.1",
    author="charonwangg",
    author_email="charonwangg@Gmail.com",
    description="A Highly-Distangible Config Based Deep Learning Framework",

    url="https://www.charonwangg.com/project/shallowmind/",

    python_requires='>=3.9',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
)