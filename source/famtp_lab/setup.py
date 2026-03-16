"""Setuptools entrypoint for famtp_lab editable installation."""

from setuptools import find_packages, setup

setup(
    name="famtp_lab",
    version="0.1.0",
    description="FaMTP Isaac Lab external-project package.",
    author="famtp-isaaclab contributors",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["gymnasium", "numpy", "torch"],
    python_requires=">=3.10",
    zip_safe=False,
)
