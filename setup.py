from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="survkit",
    version="0.0.1",
    description="Survival analysis modeling",
    long_description=readme,
    author="Todd Morrill",
    author_email="todd@cs.columbia.edu",
    url="NA",
    license=license,
    packages=find_packages(),
)