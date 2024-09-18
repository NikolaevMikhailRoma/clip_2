from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="clip-brand-recognition",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A brand recognition system using a fine-tuned CLIP model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/model_clip_2",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "torchvision",
        "transformers",
        "pillow",
        "numpy",
        "tqdm",
    ],
    extras_require={
        "dev": ["pytest", "flake8", "black", "wheel", "setuptools", "twine"],
    },
)