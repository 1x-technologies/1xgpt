from setuptools import find_packages, setup

setup(
    name="onexgpt",
    version="1.1",
    packages=find_packages(include=['genie', 'magvit2', 'genie.*', 'magvit2.*']),
)
