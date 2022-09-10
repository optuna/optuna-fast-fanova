import os

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_ext import build_ext


class LazyImportBuildExt(build_ext):
    def finalize_options(self) -> None:
        from Cython.Build import cythonize

        self.extensions = cythonize(self.extensions)
        super().finalize_options()

    def run(self) -> None:
        import numpy

        self.include_dirs.append(numpy.get_include())
        super().run()


if __name__ == "__main__":
    setup(
        packages=find_packages(exclude=("tests", "tests.*")),
        ext_modules=[
            Extension(
                "optuna_fast_fanova._fanova",
                sources=[os.path.join("optuna_fast_fanova", "_fanova.pyx")],
                language="c",
            )
        ],
        cmdclass={"build_ext": LazyImportBuildExt},
        include_package_data=False,
        package_data={"optuna_fast_fanova": ["*.pyx"]},
    )
