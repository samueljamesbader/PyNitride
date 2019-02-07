from pynitride.visual import  log, sublog
from pynitride.solvers import PoissonSolver, Equilibrium, SelfConsistentLoop
from pynitride.carriers import Schrodinger, Semiclassical, MultibandKP
from pynitride.thermal import ConstantT
from pynitride.strain import Pseudomorphic
from operator import itemgetter
from inspect import signature
from time import time
import os.path

class Simulation():

    def __init__(self,name,define_mesh,solve_flow,
                 mesh_opts={},solve_opts={},outdir=""):
        self.name=name
        self._outdir=outdir
        self.dmeshes={}
        self.rmeshes={}
        self.extras ={}

        self._define_mesh=define_mesh
        self._solve_flow=solve_flow

        self._mesh_opts=mesh_opts
        self._solve_opts=solve_opts

    @staticmethod
    def flow_semiclassicalramp_schrodinger(sim,ramp_opts={},schro_opts={},loop_opts={}):
        m,quantum,semi=sim.dmeshes['main'],sim.dmeshes['schro'],sim.dmeshes['semi']

        # General solvers
        Equilibrium(m)
        ConstantT(m)
        Pseudomorphic(m)
        ps=PoissonSolver(m)

        # Which carriers will be covered by Schrodinger and which won't
        schro_carriers=schro_opts.get('carriers',
            signature(Schrodinger).parameters['carriers'].default)
        non_schro_carriers=list(set(['electron','hole'])-set(schro_carriers))

        # Semiclassical to do at first what Schrodinger will do later
        semi_solver=Semiclassical(quantum,carriers=schro_carriers)

        # Initial ramp
        scl=SelfConsistentLoop(
            fieldsolvers=[PoissonSolver(m)],
            carriermodels=[semi_solver,
                           Semiclassical(quantum,carriers=non_schro_carriers),
                           Semiclassical(semi)])
        scl.ramp_epsfactor(**ramp_opts)

        with sublog("Schrodinger loop"):
            starttime=time()

            # Put in Schrodinger and loop again
            sim.extras['schro']=schro=Schrodinger(quantum,**schro_opts)
            scl.swap_carrier_model(remove=semi_solver,add=schro)
            scl.loop(**loop_opts)

            endtime=time()
            log("Schrodinger loop took {:.1f} sec".format(endtime-starttime))
        #log("Saving output to "+sim._outdir)

    @staticmethod
    def flow_fixedschrodinger(sim):
        pass

    @staticmethod
    def flow_semiclassicalramp_mbkp(sim,ramp_opts={},mbkp_opts={},loop_opts={}):
        m,quantum,semi=sim.dmeshes['main'],sim.dmeshes['mbkp'],sim.dmeshes['semi']

        # General solvers
        Equilibrium(m)
        ConstantT(m)
        Pseudomorphic(m)
        ps=PoissonSolver(m)

        # Which carriers will be covered by MBKP and which won't
        mbkp_carriers=mbkp_opts.get('carriers',
             signature(MultibandKP).parameters['carriers'].default)
        non_mbkp_carriers=list(set(['electron','hole'])-set(mbkp_carriers))

        # Semiclassical to do at first what MBKP will do later
        semi_solver=Semiclassical(quantum,carriers=mbkp_carriers)

        # Initial ramp
        scl=SelfConsistentLoop(
            fieldsolvers= [PoissonSolver(m)],
            carriermodels=[semi_solver,
                           Semiclassical(quantum,carriers=non_mbkp_carriers),
                           Semiclassical(semi)])
        scl.ramp_epsfactor(**ramp_opts)

        with sublog("MBKP loop"):
            starttime=time()

            # Put in MBKP and loop again
            sim.extras['mbkp']=mbkp=MultibandKP(quantum,rmesh=sim.rmeshes['mbkp'],**mbkp_opts)
            scl.swap_carrier_model(remove=semi_solver,add=mbkp)
            scl.loop(**loop_opts)

            endtime=time()
            log("MBKP loop took {:.1f} sec".format(endtime-starttime))
        #log("Saving output to "+sim._outdir)


    @staticmethod
    def flow_semiclassicalramp_mbkp_refinedmbkp(sim):
        pass

    @staticmethod
    def loader_standard(sim):
        with sublog("Hoping to load previous run from " + sim._outdir+sim.name+"*"):
            try:
                if 'main'  in sim.dmeshes:
                    sim.dmeshes['main'] .read(os.path.join(sim._outdir, sim.name+"_direct.npz"))
                if 'rmesh' in sim.rmeshes:
                    sim.rmeshes['rmesh'].read(os.path.join(sim._outdir, sim.name+"_reciprocal.npz"))
            except Exception as e:
                log("But "+str(e))
                return False
            log("Loaded")
            return True


    def load(self,force=False):
        self._define_mesh(self,**self._mesh_opts)
        loaded=(force is False) and (self._outdir is not None) and Simulation.loader_standard(self)
        if not loaded:
            with sublog("No previous solve loaded, so starting solve:"):
                self._solve_flow(self,**self._solve_opts)
                log("Done solve flow")

