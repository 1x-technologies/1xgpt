from setuptools import find_namespace_packages, setup

setup(
    name="onexgpt",
    version="1.0",
    packages=find_namespace_packages(include=['genie.*', 'magvit2.*']),
)
