# PyNitride


## Installation

If you don't have Python 3, I recommend you install the `miniconda` distribution for Python 3 from  [Continuum](<https://conda.io/miniconda.html>) following their instructions.

*Hopefully, the next steps I will replace with a conda package... but I still have to figure that out*, so for now:

Install the various Python dependencies of this project.  If you have `miniconda` or `anaconda`, then this is easy.  Open up a terminal and type:

    conda install numpy scipy cython anaconda matplotlib pytest

    conda install --channel https://conda.anaconda.org/acellera pint


Then download this project via `github`

    cd DIRECTORY_WHERE_I_WANT_TO_INSTALL
    git clone https://github.com/samueljamesbader/PyNitride.git
    
And build it

    cd PyNitride
    python setup.py build_ext -i
    
Make sure this directory is on your Python path, and you should be good to go.
