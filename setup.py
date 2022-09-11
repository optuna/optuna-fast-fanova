import os

import Cython.Build
import numpy
from setuptools import Extension
from setuptools import setup


if __name__ == "__main__":
    setup(
        ext_modules=[
            Extension(
                "optuna_fast_fanova._fanova",
                sources=[os.path.join("optuna_fast_fanova", "_fanova.pyx")],
                include_dirs=[numpy.get_include()],
                language="c",
            )
        ],
        cmdclass={"build_ext": Cython.Build.build_ext},
        package_data={"optuna_fast_fanova": ["*.pyx"]},
    )
