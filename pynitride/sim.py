from pynitride.visual import  log, sublog
from operator import itemgetter
import os.path

class Simulation():

    def __init__(self,name,define_mesh,solve_flow,meshopts={},solveopts={},outdir=""):
        self.name=name
        self._outdir=outdir
        self.dmeshes={}
        self.rmeshes={}
        self.extras ={}

        self._define_mesh=define_mesh
        self._solve_flow=solve_flow

        self._default_meshopts=meshopts
        self._default_solveopts=solveopts

    @staticmethod
    def flow_semiclassicalramp(sim,):
        pass

    @staticmethod
    def flow_fixedschrodinger(sim):
        pass

    @staticmethod
    def flow_semiclassicalramp_mbkp_refinedmbkp(sim):
        pass

    def load(self,name, outdir="", force=False, meshopts={}, solveopts={}):
        meshes =self._define_mesh(self,**meshopts)

        m, schro, semi, rmesh_out = itemgetter('main', 'schro', 'semi', 'rmesh_out')(meshes)

        if not force:
            # Try to load from file
            with sublog("Trying to load from "+outdir):
                try:
                    m.read(os.path.join(outdir, name + "_direct.npz"))
                    #PoissonSolver.update_bands_to_potential(m)
                    rmesh_out.read(os.path.join(outdir, name + "_reciprocal.npz"))
                    log("Loaded previous run from " + outdir)
                    return meshes
                except Exception as e:
                    log(e)
                    pass

        # Otherwise redo the solve
        with sublog("Previous run not loaded, so running solve"):
            self._solve_flow(self,**solveopts)
            return meshes

