from setuptools import setup

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as fp:
    install_requires = fp.read()

setup(name='AIDA',
      version='0.0.1',
      description='Analytical Inversion and Data Assimilation (AIDA)',
      long_description=readme,
      long_description_content_type='text/markdown',
      author=['Amir Souri','Jia Jung'],
      author_email='ahsouri@gmail.com',
      license='MIT',
      packages=['aida'],
      install_requires=install_requires,
      zip_safe=False)
