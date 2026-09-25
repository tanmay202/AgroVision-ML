from setuptools import setup, find_packages

setup(
    name="agrovision-ml",
    version="2.0.0",
    description="Multi-commodity agricultural price forecasting and volatility analysis",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.6.0",
        "joblib>=1.1.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
)
