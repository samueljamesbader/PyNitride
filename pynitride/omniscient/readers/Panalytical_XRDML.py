import numpy as np
import xrayutilities as xu

def read(filename):
    angle,cps=xu.io.getxrdml_scan(filename)
    return {'angle': angle, 'cps': cps}
