from setuptools import find_packages, setup

with open("virgo_app/README.md", "r") as f:
    long_description = f.read()

setup(
    name="virgo_modules",
    version="0.0.53",
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
    install_requires=["feature-engine==1.6.1","matplotlib==3.6.3","mlflow==2.1.1","numpy==1.23.5","optuna==3.1.0","pandas==1.5.3",
                "plotly==5.15.0","rsa==4.9","scikit-learn==1.2.1","scipy==1.10.0","seaborn==0.12.2","starlette==0.22.0","statsmodels==0.13.5",
                "ta==0.10.2","yfinance==0.2.9","hmmlearn==0.3.0","boto3"],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    python_requires=">=3.9, <3.10",
)