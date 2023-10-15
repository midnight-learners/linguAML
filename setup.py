from setuptools import setup

setup(
    name="linguaml",
    version="0.0.1",
    packages=["linguaml"],
    entry_points={
        "console_scripts": [
            "linguaml = linguaml:main",
        ],
    },
)
