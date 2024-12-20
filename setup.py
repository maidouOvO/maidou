from setuptools import setup, find_packages

setup(
    name="maidou-pdf",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "PyMuPDF==1.23.26",
        "pandas==2.2.1",
        "Pillow==10.2.0",
        "fpdf==1.7.2",
    ],
    entry_points={
        'console_scripts': [
            'maidou-pdf=main:main',
        ],
    },
    author="maidouOvO",
    description="A tool for extracting and editing text from PDF files",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
