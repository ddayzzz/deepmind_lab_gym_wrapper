import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="deepmind_lab_as_gym",
    version="0.1.1",
    author="Shu Wang",
    author_email="wangshu214@live.cn",
    description="Gym interface to deepmind lab environment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ddayzzz/deepmind_lab_gym_wrapper",
    packages=setuptools.find_packages(exclude=('unittest',)),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "gym",
        "numpy",
        "opencv-python",
        "opencv-contrib-python",
        "DeepMind-Lab"
    ],
)