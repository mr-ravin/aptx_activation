from setuptools import setup, find_packages

setup(
    name="aptx_activation",
    version="0.0.1",
    author="Ravin Kumar",
    author_email="mr.ravin_kumar@hotmail.com",
    description="A PyTorch implementation of the APTx activation function.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mr-ravin/aptx_activation",
    packages=find_packages(),
    install_requires=[
        "torch>=1.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords=[
        "APTx", "activation function", "deep learning", "pytorch", "neural networks",
        "machine learning", "AI", "torch", "MISH", "SWISH", "ReLU", "aptx", "ML", "AI", "DL"
    ],
    license="MIT"
)
