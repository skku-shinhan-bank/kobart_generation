from setuptools import setup, find_packages

with open('requirements.txt', "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setup(
    name="skku-shinhan-bank/kobart_generation",
    version="0.0.1",
    author="SKKU Shinhan Bank",
    author_email="ajtwlswjddnv1102@gmail.com",
    description="kobart generation",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/skku-shinhan-bank/kobart_generation",
    packages=find_packages(),
    install_requires=requirements,
)