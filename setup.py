import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transmatching",  # Replace with your own username
    version="0.0.1",
    author="Example Author",
    author_email="author@example.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "tqdm",
        "trimesh",
        "matplotlib",
        "plotly",
        "pytorch-lightning",
        "dvc[gdrive]==2.13.0",
        "meshio==5.3.4",
        "scipy==1.8.1",
        "scikit-learn==1.1.1",
        "python-dotenv==0.20.0",
        "hydra-core==1.1",
        "GitPython==3.1.27",
        "streamlit==1.10.0",
        "stqdm==0.0.4",
    ],
)
