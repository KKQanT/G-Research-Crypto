from setuptools import setup, find_packages

install_requires = []
with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

packages = find_packages()

setup(
    name='g_research_crypto',
    version='1.0.0',
    description='',
    author='KKQanT',
    author_email="asskarnwin@gmail.com",
    packages=packages,
    install_requires=install_requires,
    url='https://github.com/KKQanT/G-Research-Crypto'
)