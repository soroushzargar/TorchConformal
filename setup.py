import setuptools


with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name="torchconformal",
    version="0.1.0",

    author='Soroush H. Zargarbashii',
    author_email='soroushzargar@gmail.com',
    url='https://github.com/soroushzargar/TorchConformal.git',

    description="Pytorch Impelmentation of Conformal Prediction",

    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=setuptools.find_packages(),
    install_requires=[
        'seaborn',
        'ml-collections==0.1.1',
        'sacred',
        'statsmodels==0.14.1',
    ],

    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    license='License :: OSI Approved :: MIT License',
)
