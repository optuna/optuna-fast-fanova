import os

from Cython.Build import cythonize
import numpy
from setuptools import Extension
from setuptools import find_packages
from setuptools import setup


if __name__ == "__main__":
    setup(
        packages=find_packages(exclude=("tests", "tests.*")),
        ext_modules=cythonize(
            [
                Extension(
                    "optuna_fast_fanova._fanova",
                    sources=[os.path.join("optuna_fast_fanova", "_fanova.pyx")],
                    include_dirs=[numpy.get_include()],
                    language="c",
                )
            ]
        ),
        include_package_data=False,
        package_data={"optuna_fast_fanova": ["*.pyx"]},
    )
