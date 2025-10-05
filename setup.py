from setuptools import find_packages, setup

with open("virgo_app/README.md", "r") as f:
    long_description = f.read()

setup(
    name="virgo_modules",
    version="0.8.2",
    description="data processing and statistical modeling using stock market data",
    package_dir={"": "virgo_app"},
    packages=find_packages(where="virgo_app"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/miguelmayhem92/virgo_module",
    author="Miguel Mayhuire",
    author_email="miguelmayhem92@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    python_requires=">=3.9",
)