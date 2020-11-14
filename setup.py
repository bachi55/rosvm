from setuptools import setup, find_packages

setup(
    name="rosvm",
    version="0.4.0",
    license="MIT",
    packages=find_packages(exclude=["results*", "tests", "examples", "*.ipynb", "run_*.py"]),

    # Minimum requirements the package was tested with
    install_requires=[
        "numpy",
        "scikit-learn",
        "pandas",
        "scipy",
        "joblib",
        "cvxpy",
        "rdkit"
    ],

    # Metadata
    author="Eric Bach",
    author_email="eric.bach@aalto.fi",
    description="RankSVM implementation to predict the retention order of molecular structures.",
    url="https://github.com/bachi55/rosvm",
)
