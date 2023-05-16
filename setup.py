from setuptools import setup

setup(
    name='cnn_architectures',
    version='1.1',
    packages=['cnn_architectures'],
    url='https://github.com/mmunar97/cnn_architectures',
    license='mit',
    author='marcmunar',
    author_email='marc.munar@uib.es',
    description='Implementation of some architectures of CNN for binary segmentation written in TensorFlow',
    include_package_data=True,
    install_requires=[
        "tensorflow==2.3.0",
        "numpy",
        "opencv-python",
        "scikit-image"
    ]
)