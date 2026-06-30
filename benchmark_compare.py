import time
import numpy as np
from qutip import sigmax, sigmaz, basis, expect
from qutip.solver.heom import HEOMSolver, DrudeLorentzBath

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    comm = None
    rank = 0
    size = 1

def print0(*args):
    if rank == 0:
        print(*args, flush=True)

def run_benchmark():
    # ------------------------------------------------------------------
    # 1. System Parameters
    # ------------------------------------------------------------------
    epsilon = 0.5
    delta = 1.0
    H = epsilon/2 * sigmaz() + delta/2 * sigmax()
    Q = sigmaz()
    rho0 = basis(2,0) * basis(2,0).dag()

    # The "Light" Benchmark parameters (Safe for 8GB RAM for the 'csr' backend)
    Nk = 6
    MAX_DEPTH = 12
    
    bath = DrudeLorentzBath(
        Q,
        lam=0.1,
        gamma=0.5,
        T=300,
        Nk=Nk,
    )

    tlist = np.linspace(0, 2, 50)  # Shortened to keep testing quick

    print0(f"\n===== HEOM Backend Comparison Benchmark =====")
    print0(f"MPI Ranks = {size}")
    print0(f"Nk        = {Nk}")
    print0(f"Depth     = {MAX_DEPTH}")
    
    t_asm_csr = t_ss_csr = t_evo_csr = 0.0
    
    # ------------------------------------------------------------------
    # 2. Benchmark Standard 'csr' Backend (Rank 0 only)
    # ------------------------------------------------------------------
    if False: # rank == 0:
        print0("\n>>> 1. Running Standard 'csr' Backend (Single Node) <<<")
        try:
            # Assembly
            t0 = time.time()
            solver_csr = HEOMSolver(
                H, [bath], max_depth=MAX_DEPTH, 
                options={"backend": "csr", "store_ados": False, "progress_bar": None}
            )
            t_asm_csr = time.time() - t0
            print0(f"   [csr] Assembly Time      : {t_asm_csr:.4f} s")
            print0(f"   Total ADOs               : {len(solver_csr.ados.labels)}")
            
            # Steady State
            t0 = time.time()
            rho_ss_csr, _ = solver_csr.steady_state(use_mkl=True)
            t_ss_csr = time.time() - t0
            print0(f"   [csr] Steady State Time  : {t_ss_csr:.4f} s")
            
            # Time Evolution
            t0 = time.time()
            res_csr = solver_csr.run(rho0, tlist, e_ops=[rho0])
            t_evo_csr = time.time() - t0
            print0(f"   [csr] Evolution Time     : {t_evo_csr:.4f} s")
            
            # Force Memory cleanup before starting PETSc
            del solver_csr
            import gc; gc.collect()
        except Exception as e:
            print0(f"   [csr] Failed: {e}")

    # Synchronize all ranks before starting PETSc
    if comm is not None:
        comm.Barrier()

    # ------------------------------------------------------------------
    # 3. Benchmark Distributed 'petsc' Backend (All Ranks)
    # ------------------------------------------------------------------
    print0(f"\n>>> 2. Running Distributed 'petsc' Backend (Across {size} Ranks) <<<")
    try:
        # Assembly
        t0 = time.time()
        solver_petsc = HEOMSolver(
            H, [bath], max_depth=MAX_DEPTH, 
            options={"backend": "petsc", "store_ados": False, "progress_bar": None}
        )
        if comm is not None: comm.Barrier()
        t_asm_petsc = time.time() - t0
        print0(f"   [petsc] Assembly Time    : {t_asm_petsc:.4f} s")
        
        # Steady State
        t0 = time.time()
        rho_ss_petsc, _ = solver_petsc.steady_state(ksp_type="bcgs", pc_type="jacobi")
        if comm is not None: comm.Barrier()
        t_ss_petsc = time.time() - t0
        print0(f"   [petsc] Steady State Time: {t_ss_petsc:.4f} s")
        
        # Time Evolution
        t0 = time.time()
        res_petsc = solver_petsc.run(rho0, tlist, e_ops=[rho0])
        if comm is not None: comm.Barrier()
        t_evo_petsc = time.time() - t0
        print0(f"   [petsc] Evolution Time   : {t_evo_petsc:.4f} s")
        
        # Final Verification
        if rank == 0:
            print0("\n>>> 3. Results Comparison <<<")
            if t_asm_csr > 0:
                print0(f"   Speedup (Assembly)     : {t_asm_csr / t_asm_petsc:.2f}x")
                print0(f"   Speedup (Steady State) : {t_ss_csr / t_ss_petsc:.2f}x")
                print0(f"   Speedup (Evolution)    : {t_evo_csr / t_evo_petsc:.2f}x")
            
    except Exception as e:
        import traceback
        if rank == 0:
            traceback.print_exc()
        print0(f"   [petsc] Failed: {e}")
        
    print0("\n===== BENCHMARK COMPLETE =====")

    ts = solver_petsc._integrator.ts

    if rank == 0:
        print("TS Type:", ts.getType())
        print("TS Steps:", ts.getStepNumber())
        print("Final dt:", ts.getTimeStep())
        print("SNES Iterations:", ts.getSNESIterations())
        print("KSP Iterations:", ts.getKSPIterations())

if __name__ == "__main__":
    run_benchmark()
