"""
Setup configuration for Estimation Bot package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="estimation-bot",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI bot for Middle Eastern Estimation card game",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/estimation-bot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Games/Entertainment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pytest>=6.0.0",
    ],
    extras_require={
        "ml": [
            "torch>=1.9.0",
            "gym>=0.21.0",
        ],
        "analysis": [
            "pandas>=1.3.0",
            "matplotlib>=3.4.0",
        ],
        "dev": [
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "estimation-game=main:main",
            "estimation-train=bot.trainer:run_quick_evaluation",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)