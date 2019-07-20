from setuptools import setup
from setuptools import find_packages


setup(
    name='WMDO',
    version='1.0.0',
    description='WMDO',
    packages=find_packages(),
    zip_safe=False,
    entry_points={
        "console_scripts": [
            'compute-wmdo=WMDO.bin.compute_wmdo:main'
        ]
    }
)
