from setuptools import setup, find_packages
import os

# 获取当前文件的路径
current_directory = os.path.abspath(os.path.dirname(__file__))

# 读取README.md的内容
with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='MedEvalKit',
    version='0.1.0',
    author='HanRu1',
    author_email='2053243548@qq.com',
    description='A Python package for medical image metrics and operations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/HanRu1/MedEvalKit.git',
    packages=find_packages(),

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='medical imaging metrics numpy scipy skimage sklearn',
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'scipy',
        'SimpleITK',
        'scikit-image',
        'scikit-learn',
        'sewar',
        'torch',
        'torchvision',
        'nibabel',
    ],
    include_package_data=True,
    zip_safe=False
)
