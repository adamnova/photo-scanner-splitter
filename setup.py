from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="photo-scanner-splitter",
    version="0.1.0",
    author="Adam Nova",
    description="A tool to split and align scanned photos into individual images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adamnova/photo-scanner-splitter",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "photo-splitter=photo_splitter.cli:main",
        ],
    },
)
