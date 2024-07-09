from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> List[str]:
    """return a list of requirements"""

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [r.replace("\n", "") for r in requirements]
    
    if "-e ." in requirements:
        requirements.remove("-e .")
        
    return requirements
    

setup(
    name="ml-project",
    version="0.0.1",
    author="tuoling",
    author_email="qiutuoling8888@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)