from pynitride import  log, sublog, to_unit
from pynitride import PoissonSolver, Equilibrium, SelfConsistentLoop, Linear_Fermi
from pynitride import Schrodinger, Semiclassical, MultibandKP
from pynitride import ConstantT, Pseudomorphic
from inspect import signature
from time import time
import os.path
import pickle
from pathlib import Path

class Simulation():

    def __init__(self,name,define_mesh,solve_flow,
                 mesh_opts={},solve_opts={},extras=[],outdir=""):
        """ Manages the running/re-loading of a simulation .

        Breaks apart a simulation into the "mesh definition" which is a rapid step executed on both running and
        re-loading of simulations and "solve flow' which is the slow actual computation.

        Various `define_mesh` functions can be found under :mod:`pynitride.examples`, and various `solve_flow`
        functions are built into this class.  `define_mesh` should place meshes and rmeshes into the
        `dmeshes` and `rmeshes` dict, and may place other free-form information into `extras`.  `solve_flow` functions
        specify which the necessary keys/content are for these dicts.

        Args:
            name: Name for the simulation (becomes part of the relevant filenames)
            define_mesh: function which defines the mesh, first argument should be this `Simulation` object)
            solve_flow: function executes the solve flow, first argument should be this `Simulation` object)
            mesh_opts: additional arguments to be passed to `define_mesh`
            solve_opts: additional arguments to be passed to `solve_flow`
            extras: name of extra objects saved by the `solve_flow` which should be loaded
            outdir: directory where to save/read the results
        """
        self.name=name
        """ Name of the simulation. """

        self._outdir=outdir
        """ Where to put the output"""
        if outdir and len(outdir): Path(outdir).mkdir(parents=True,exist_ok=True)

        self.dmeshes={}
        """ Where the (direct space) meshes are stored. """

        self.rmeshes={}
        """ Where the reciprocal space meshes are stored. """

        self.extras ={}
        """ Where `define_mesh` provides any further information. """

        self._extras =extras

        self._define_mesh=define_mesh
        self._solve_flow=solve_flow

        self._mesh_opts=mesh_opts
        self._solve_opts=solve_opts

    @staticmethod
    def flow_semiclassicalramp(sim,ramp_opts={}):
        """ Does a ramp with semiclassical solver.

        The main mesh should be `dmeshes['main']`.

        Args:
            sim: the Simulation object (ie `self`)
            ramp_opts: passed to the :func:`~pynitride.physics.solvers.SelfConsistentLoop.ramp_epsfactor`

        """
        m=sim.dmeshes['main']

        # General solvers
        Equilibrium(m)
        ConstantT(m).solve()
        Pseudomorphic(m).solve()
        ps=PoissonSolver(m)
        semi_solver=Semiclassical(m)

        # Initial ramp
        scl=SelfConsistentLoop(
            fieldsolvers=[PoissonSolver(m)],
            carriermodels=[semi_solver])
        scl.ramp_epsfactor(**ramp_opts)
    @staticmethod
    def flow_semiclassicalramp_schrodinger(sim,ramp_opts={},schro_opts={},loop_opts={}):
        """ Does a ramp with semiclassical solver, then swaps in a schrodinger solver in dmeshes['schro'] region.

        The main mesh should be `dmeshes['main']`, the schrodinger region should be `dmeshes['schro']`, and the
        semiclassical region should be `dmeshes['semi']`.

        Args:
            sim: the Simulation object (ie `self`)
            ramp_opts: passed to the :func:`~pynitride.physics.solvers.SelfConsistentLoop.ramp_epsfactor`
            schro_opts: passed to the :class:`~pynitride.physics.carriers.Schrodinger` solver
            loop_opts: passed to the self-consistent :func:`~pynitride.physics.solvers.SelfConsistentLoop.loop`

        """
        m,quantum,semi=sim.dmeshes['main'],sim.dmeshes['schro'],sim.dmeshes['semi']

        # General solvers
        Equilibrium(m)
        ConstantT(m).solve()
        Pseudomorphic(m).solve()
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
    def flow_semiclassicalramp_mbkp(sim,T=300,Va=0,strain=None,ramp_opts={},mbkp_opts={},loop_opts={},mbkp_loop_opts={}):
        """ Does a ramp with semiclassical solver, then swaps in an MBKP solver in dmeshes['mbkp'] region.

        The main mesh should be `dmeshes['main']`, the quantum region should be `dmeshes['mbkp']`, and the
        semiclassical region should be `dmeshes['semi']. `extras['sourcepoint']` should be the interface or z-coordinate
        (as specified for a :class:`~pynitride.solvers.LinearFermi` contact) of the point in the structure held to zero,
        eg location of a source-connected carrier gas.

        Args:
            sim: the Simulation object (ie `self`)
            T: the temperature
            Va: the applied voltage
            strain: passed to :class:`pynitride.physics.strain.Pseudomorphic`
            ramp_opts: passed to the :func:`~pynitride.physics.solvers.SelfConsistentLoop.ramp_epsfactor`
            mbkp_opts: passed to the :class:`~pynitride.physics.carriers.MultibandKP` solver
            loop_opts: passed to the self-consistent :func:`~pynitride.physics.solvers.SelfConsistentLoop.loop`
                when called for the ramp
            mbkp_loop_opts: passed to the self-consistent :func:`~pynitride.physics.solvers.SelfConsistentLoop.loop`
                when called for MBKP
        """
        m,quantum,semi=sim.dmeshes['main'],sim.dmeshes['mbkp'],sim.dmeshes['semi']

        # General solvers
        lf=Linear_Fermi(m,dict(gate=0,hg=sim.extras['sourcepoint'],subs=-1))
        lf.solve(gate=Va)
        ConstantT(m,T).solve()
        Pseudomorphic(m,straincond=strain).solve()
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
            scl.loop(**mbkp_loop_opts)

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
        """ A simple loader for typical names, don't call directly, use :func:`Simulation.load`."""
        with sublog("Hoping to load previous run from " + os.path.join(sim._outdir,sim.name+"*")):
            try:
                m=sim.dmeshes.get('main',False)
                if m:
                    m.read(os.path.join(sim._outdir, sim.name+"_direct.npz"))
                rmesh=sim.rmeshes.get('mbkp_out',False) or sim.rmeshes.get('mbkp',False) or sim.rmeshes.get('mbkp_solve',False)
                if rmesh:
                    rmesh.read(os.path.join(sim._outdir, sim.name+"_reciprocal.npz"))
                for extra in sim._extras:
                    with open(os.path.join(sim._outdir, sim.name+"_"+extra+".pkl"),'rb') as f:
                        sim.extras[extra]=pickle.load(f)
            except Exception as e:
                log("But "+str(e))
                return False
            return True


    def load(self,force=False):
        """ Loads the simulation or re-runs it if not able to find/load.

        Args:
            force: whether to force a fresh run even if a previous one can be found
        """
        self._define_mesh(self,**self._mesh_opts)
        loaded=(force is False) and (self._outdir is not None) and Simulation.loader_standard(self)
        if loaded:
            log("Loaded")
        else:
            with sublog("Starting solve:"):
                self._solve_flow(self,**self._solve_opts)
                log("Done solve flow")

