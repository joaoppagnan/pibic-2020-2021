from setuptools import setup, find_packages

VERSION = '1.0' 
DESCRIPTION = 'Pacote de Python para o projeto do PIBIC 2020/2021'
LONG_DESCRIPTION = 'Pacote de Python para o projeto do PIBIC 2020/2021'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="pibic2020", 
        version=VERSION,
        author="João Pedro de Oliveira PagnaN",
        author_email="<j199727@dac.unicamp.br>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 1.0",
            "Intended Audience :: Pesquisadores e Estudantes",
            "Programming Language :: Python :: 3",
            "Operating System :: Linux :: Ubuntu 18.04",
        ]
)