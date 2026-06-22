from setuptools import setup, find_packages

setup(
    name            = "rag-hallucination-detector",
    version         = "1.0.0",
    description     = "DeBERTa-v3 fine-tuned on RAGTruth for RAG hallucination detection",
    author          = "Your Name",
    python_requires = ">=3.10",
    packages        = find_packages(where="src"),
    package_dir     = {"": "src"},
    install_requires=[
        "torch>=2.1.0",
        "transformers==4.53.0",
        "datasets==3.6.0",
        "accelerate==1.8.1",
        "evaluate==0.4.5",
        "scikit-learn==1.7.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.7.0", "seaborn>=0.12.0"],
        "dev": ["pytest", "black", "isort", "flake8"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
