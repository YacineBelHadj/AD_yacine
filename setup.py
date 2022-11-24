from setuptools import find_packages, setup

setup(
    name='AD_vibration',
    packages=find_packages(include=['AD_vibration', 'AD_vibration.*']),
    version='0.1.0',
    description='A project to perform anomaly detection on structural vibration',
    author='Yacine Bel-Hadj',
    license='MIT',
)
