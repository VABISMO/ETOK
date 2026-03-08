from setuptools import setup, find_packages

setup(
    name="etok",
    version="0.1.0",
    description="BPE tokenizer with magic_split, rotate_compare, and entropy-adaptive training",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Vicent Nos Ripolles",
    author_email="vicent@cobalt-technologies.pa",
    url="https://github.com/cobalt-technologies-pa/etok",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    python_requires=">=3.8",
    install_requires=[],   # zero dependencies
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
    ],
    keywords="tokenizer BPE NLP morphology entropy",
)
