from pynitride.visual import  log, sublog
from pynitride.solvers import PoissonSolver, Equilibrium, SelfConsistentLoop
from pynitride.carriers import Schrodinger, Semiclassical, MultibandKP
from pynitride.thermal import ConstantT
from pynitride.strain import Pseudomorphic
from pynitride.paramdb import to_unit
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
    def flow_semiclassicalramp_mbkp(sim,T=300,strain=None,ramp_opts={},mbkp_opts={},loop_opts={}):
        m,quantum,semi=sim.dmeshes['main'],sim.dmeshes['mbkp'],sim.dmeshes['semi']

        # General solvers
        Equilibrium(m)
        ConstantT(m,T)
        Pseudomorphic(m,straincond=strain)
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

        # MBKP loop
        rmesh=sim.rmeshes['mbkp_solve'] if 'mbkp_solve' in sim.rmeshes else sim.rmeshes['mbkp']
        with sublog("MBKP loop"):
            starttime=time()

            # Put in MBKP and loop again
            sim.extras['mbkp']=mbkp=MultibandKP(quantum,rmesh=rmesh,**mbkp_opts)
            scl.swap_carrier_model(remove=semi_solver,add=mbkp)
            scl.loop(**loop_opts)

            endtime=time()
            log("MBKP loop took {:.1f} sec".format(endtime-starttime))

        # Refinement
        if 'mbkp_out' in sim.rmeshes:
            starttime=time()
            log("Refining MBKP")

            # Refine k-space
            rmesh=sim.rmeshes['mbkp_out']
            sim.extras['mbkp'] =mbkp= MultibandKP(quantum, num_eigenvalues=6, rmesh=rmesh)
            mbkp.solve()

            endtime=time()
            log("MBKP refinement took {:.1f} sec".format(endtime-starttime))

        # Useful checks
        log("Holes: {:.2f} x10^13/cm^2".format(
            to_unit(float(m.p.integrate(definite=True)), "1e13/cm^2")))
        log("EV-EF [meV]: {:.2f} meV".format(
            to_unit(float((m.Ev-m.EF.tmf())[m.indexm(sim.extras['well_t'])]),"meV")))

        # Save output
        log("Saving to: " + sim._outdir)
        m.save(os.path.join(sim._outdir, sim.name + "_direct"), )
        rmesh.save(os.path.join(sim._outdir, sim.name + "_reciprocal"), ['kpen'])

    @staticmethod
    def loader_standard(sim):
        with sublog("Hoping to load previous run from " + os.path.join(sim._outdir,sim.name+"*")):
            try:
                m=sim.dmeshes.get('main',False)
                if m:
                    m.read(os.path.join(sim._outdir, sim.name+"_direct.npz"))
                rmesh=sim.rmeshes.get('mbkp_out',False) or sim.rmeshes.get('mbkp',False)
                if rmesh:
                    rmesh.read(os.path.join(sim._outdir, sim.name+"_reciprocal.npz"))
            except Exception as e:
                log("But "+str(e))
                return False
            return True


    def load(self,force=False):
        self._define_mesh(self,**self._mesh_opts)
        loaded=(force is False) and (self._outdir is not None) and Simulation.loader_standard(self)
        if loaded:
            log("Loaded")
        else:
            with sublog("Starting solve:"):
                self._solve_flow(self,**self._solve_opts)
                log("Done solve flow")

