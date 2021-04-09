from setuptools import setup, find_packages
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('rosvm/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name="rosvm",
    version=main_ns["__version__"],
    license="MIT",
    packages=find_packages(exclude=["results*", "tests", "examples", "*.ipynb", "run_*.py"]),

    # Minimum requirements the package was tested with
    install_requires=[
        "numpy",
        "scikit-learn",
        "pandas",
        "scipy",
        "joblib",
        "cvxpy"
    ],

    # Metadata
    author="Eric Bach",
    author_email="eric.bach@aalto.fi",
    description="RankSVM implementation to predict the retention order of molecular structures.",
    url="https://github.com/bachi55/rosvm",
)
