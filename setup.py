#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Jephte Abijuru",
    author_email='abijuru@rptu.de',
    python_requires='>=3.7',
    classifiers=[

    ],
    description="CEMTOM",
    entry_points={

    },
    install_requires=requirements,
    license="",
    long_description=readme,
    include_package_data=True,
    keywords='cemtom',
    name='cemtom',
    packages=find_packages(include=['cemtom', 'cemtom.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='',
    version='0.0.1',
    zip_safe=False,
)
