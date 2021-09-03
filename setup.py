import os

from setuptools import find_packages, setup

with open(os.path.join(".", "requirements.txt")) as f:
    install_deps = [
        line.strip() if not line.startswith("--editable") else ""
        for line in f.readlines()
    ]

setup(
    name="deepforge-redshift",
    version="0.0.1",
    description="Redshift Project's operation and pipelines implemented in python",
    author="Umesh Timalsina",
    author_email="umesh.timalsina@vanderbilt.edu",
    install_requires=install_deps,
    license="Apache-2.0",
    packages=find_packages(),
    zip_safe=True,
    include_package_data=True,
)
