# PyNitride

## Requirements

For now, the parallelized Python code only works on *nix systems, so Windows users will have to set the configuration
limit parallelization to C extensions only.

## Installation

If you don't have Python 3, I recommend you install the `miniconda` distribution for Python 3 from  [Continuum](<https://conda.io/miniconda.html>) following their instructions.

*Hopefully, the next steps I will replace with a conda package... but I still have to figure that out*, so for now:

Install the various Python dependencies of this project.  If you have `miniconda` or `anaconda`, then this is easy.  Open up a terminal and type:

    conda install numpy scipy cython anaconda matplotlib pytest

    conda install -c conda-forge pint
    
    conda install -c samjbader omniscient

Then download this project via `github`

    cd DIRECTORY_WHERE_I_WANT_TO_INSTALL
    git clone https://github.com/samueljamesbader/PyNitride.git
    
And build it

    cd PyNitride
    python setup.py build_ext -i
    
Make sure this directory is on your Python path (eg by running `python setup.py develop`), and you should be good to go.
