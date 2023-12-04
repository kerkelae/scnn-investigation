from setuptools import setup

setup(
    name="scnn",
    version="0.1",
    author="Leevi Kerkelä",
    author_email="leevi.kerkela@protonmail.com",
    license="MIT",
    packages=["scnn"],
    install_requires=[
        "dipy",
        "healpy",
        "matplotlib",
        "nibabel",
        "numpy",
        "scipy",
        "seaborn",
        "sympy",
        "torch",
    ],
)
