from setuptools import setup, find_packages

VERSION = "0.1.0"
PACKAGE_NAME = "carefree-workflow"

DESCRIPTION = "Build arbitray workflows with Python!"
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

openai_requires = [
    "openai",
]
ai_requires = [
    "facexlib",
    "carefree-learn[cv_full]>=0.5.0",
]
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    entry_points={"console_scripts": ["cflow = cflow.cli:main"]},
    install_requires=[
        "pillow",
        "aiohttp",
        "fastapi",
        "uvicorn",
        "networkx",
        "requests",
        "matplotlib",
        "websockets",
        "opencv-python-headless",
        "click>=8.1.3",
        "carefree-toolkit>=0.3.11",
    ],
    extras_require={
        "ai": ai_requires,
        "openai": openai_requires,
        "full": openai_requires + ai_requires,
    },
    author="carefree0910",
    author_email="syameimaru.saki@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords="python workflow",
)
