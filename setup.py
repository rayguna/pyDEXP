from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pyDEXP",
    version="1.0.0",
    author="Ray Gunawidjaja",
    author_email="ray.gunaw@gmail.com",
    description="Generate design matrix for design of experiments (DOE).",
    long_description="Capabilities include: generating design matrices for 2^k experiments, analyzing the data using Pareto chart and ANOVA, and fitting the data to a linear regression model. The current version is only limited to a 2-level full factorial design.",
    long_description_content_type="text/markdown",
    url="https://github.com/rayguna/pyDEXP",
    project_urls={
        "Bug Tracker": "https://github.com/rayguna/pyDEXP/issues",
    },
    license="MIT",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        'pandas>=1.0.0,<=1.3.3',  # Compatible with pandas 1.0.0 up to 1.3.3
        'numpy>=1.18.0,<=1.21.2',  # Compatible with numpy 1.18.0 up to 1.21.2
        'matplotlib>=3.1.0,<=3.4.3',  # Compatible with matplotlib 3.1.0 up to 3.4.3
        'scipy>=1.4.0,<=1.7.1',  # Compatible with scipy 1.4.0 up to 1.7.1
    ],

)
