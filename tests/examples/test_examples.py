import time
import numpy as np
from pathlib import Path
from typing import Callable, TextIO
from io import StringIO
from pynitride.core.sim import Simulation

from pynitride.util import returner_context

SimFunc=Callable[[],Simulation]

def create_golden_file(get_sim_func:SimFunc,example_name:str):
    sim=get_sim_func()
    sim.save_bands_file(Path(__file__).parent/f"goldens/{example_name}_bands.txt")

def read_output(file:str|Path|TextIO):
    with (open(file,'r') if isinstance(file, (str, Path)) else returner_context(file)) as f:
        headers=f.readline().strip().split(',')
        data=np.loadtxt(f,delimiter=',')
        return dict(zip(headers,data.T))

def read_golden_file(example_name:str):
    return read_output(Path(__file__).parent/f"goldens/{example_name}_bands.txt")

def _test_example(get_sim_func:SimFunc, example_name:str, create:bool=False, max_time:float=300):
    if create: create_golden_file(get_sim_func,example_name)

    golden = read_golden_file(example_name)
    start_time = time.time()
    sim=get_sim_func()
    elapsed_time = time.time() - start_time
    assert elapsed_time < max_time, f"Simulation took {elapsed_time:.2f} seconds, which exceeds the maximum allowed time of {max_time} seconds for {example_name}"
    sio = StringIO()
    sim.save_bands_file(sio)
    sio.seek(0)
    test_data = read_output(sio)

    band_precision = 1e-6
    carrier_precision = 1e8
    for key in golden.keys():
        atol = band_precision if key in ['z[nm]', 'Ec[eV]', 'Ev[eV]', 'EF[eV]']\
            else carrier_precision if key in ['n[1/cm^3]','p[1/cm^3]']\
            else None
        assert atol is not None, f"Unknown key {key} in golden file for {example_name}"
        assert np.allclose(golden[key], test_data[key], atol=atol), f"Mismatch in {key} for {example_name}"

def test_hemt_example(create:bool=False):
    from pynitride.examples.AlGaN_GaN_HEMT.hemt_example import do_simulation
    _test_example(do_simulation,"hemt_example",create=create, max_time=1)

if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        func_name=sys.argv[1]
        assert func_name.startswith("test_") and func_name in globals(),\
            "Please provide a test function name from this file starting with 'test_' as an argument"
        test_func=globals()[func_name]
        test_func(create=True)
    else:
        test_hemt_example()