import os

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_ext import build_ext


try:
    from Cython.Build import cythonize

    ext = ".pyx"
except ImportError:
    cythonize = None
    ext = ".c"


class LazyImportBuildExt(build_ext):
    def finalize_options(self) -> None:
        # cythoinze() must be lazily called since Cython's build requires scikit-learn.
        if cythonize is not None:
            self.extensions = cythonize(self.extensions)
        super().finalize_options()

    def run(self) -> None:
        import numpy

        self.include_dirs.append(numpy.get_include())
        super().run()


def get_long_description() -> str:
    readme_filepath = os.path.join(os.path.dirname(__file__), "README.md")
    with open(readme_filepath) as f:
        return f.read()


def get_version() -> str:
    version_filepath = os.path.join(os.path.dirname(__file__), "optuna_fast_fanova", "__init__.py")
    with open(version_filepath) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split()[-1][1:-1]
    assert False, "must not reach here"


if __name__ == "__main__":
    setup(
        name="optuna-fast-fanova",
        version=get_version(),
        description="Cython accelerated fANOVA implementation for Optuna",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        author="Masashi Shibata",
        author_email="mshibata@preferred.jp",
        project_urls={
            "Source": "https://github.com/optuna/optuna-fast-fanova",
            "Bug Tracker": "https://github.com/optuna/optuna-fast-fanova/issues",
        },
        # You need to install Cython when building this package from sources.
        setup_requires=["numpy", "scikit-learn"],
        install_requires=["optuna"],
        packages=find_packages(exclude=("tests", "tests.*")),
        ext_modules=[
            Extension(
                "optuna_fast_fanova._fanova",
                sources=[os.path.join("optuna_fast_fanova", "_fanova" + ext)],
                language="c",
            )
        ],
        cmdclass={"build_ext": LazyImportBuildExt},
        include_package_data=False,
        package_data={"optuna_fast_fanova": ["*.c", "*.pyx"]},
        classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3 :: Only",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development",
            "Topic :: Software Development :: Libraries",
            "Topic :: Software Development :: Libraries :: Python Modules",
        ],
    )
