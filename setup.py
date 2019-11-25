import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EnsemblePursuit",
    version="0.0.2",
    author="Marius Pachitariu, Carsen Stringer, Maria Kesa",
    author_email="maria.kesa@gmail.com",
    description="A sparse matrix factorization algorithm for extracting co-activating neurons from large-scale recordings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MouseLand/EnsemblePursuit",
    packages=setuptools.find_packages(),
	install_requires = ['numpy>=1.13.0', 'scipy', 'scikit-learn', 'torch'],
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ),
    python_requires='>=3.6',
)
