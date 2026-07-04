#!/usr/bin/env python3
"""Shared foundation for the NCTO tuned-Kruger campaign.

Single source of truth for the Hamiltonian, the *corrected* signed-Grueneisen
E1 magnetoelastic coupling, lattice geometry, reference states, order
parameters, solver invocation (OMP-pinned) and a shared parallel runner.

All campaign drivers (`run_campaign.py`) import from here so the physics
constants and the lambda couplings are defined exactly once.
"""
from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Paths / binaries.  The scripts resolve the solver checkout as their
# great-grandparent (scripts/ -> NCTO_project/ -> NCTO/ -> ClassicalSpin_Cpp/).
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "build" / "spin_solver"          # SA / MD solver
FIELD_EVAL = ROOT / "build" / "gneb_field_eval"  # per-config energy/field eval
CAMPAIGN = ROOT / "NCTO_project" / "tuned_kruger_campaign"

sys.path.insert(0, str(ROOT / "NCTO_project" / "scripts" / "np_campaign"))
sys.path.insert(0, str(ROOT / "util" / "readers_new"))
from analysis_utils import M_VECTORS  # noqa: E402
from initial_states import triple_q_state, single_zigzag_state  # noqa: E402
from reader_strain_lattice import generate_honeycomb_positions as honeycomb_positions  # noqa: E402

# --------------------------------------------------------------------------- #
# Tuned-Kruger Hamiltonian (meV) and the corrected E1 magnetoelastic coupling.
# --------------------------------------------------------------------------- #
J, K, GAMMA, GAMMAP = 0.68, -7.89, 3.07, -2.94
J2_A, J2_B, J3 = -0.06, -0.70, 0.52
J7_DEFAULT = -0.40                 # near-SU(2) switchable degeneracy (paper J7/4 = -0.10)

LAMBDA_K2_DEFAULT = 0.02
# Quadratic E1 striction:  dX_gamma(eps) = lambda_{X,2} [ (eps_x^2-eps_y^2)cos2theta
#                                                         + 2 eps_x eps_y sin2theta ].
# Uniform (isotropic) Grueneisen => common fractional modulation dX/X, i.e. the
# deformation potential scales linearly *and with sign* with the bare exchange:
#       lambda_{X,2} = (X / K) * lambda_{K,2}          (signed ratio, NOT |X|/|K|).
# The earlier |X|/|K| choice had the wrong sign on the J and Gamma channels.
ME_RATIOS = {"K": 1.0, "J": J / K, "Gamma": GAMMA / K, "Gammap": GAMMAP / K}

# Phonon / pump / MD defaults (shared by every driven run).
OMEGA_E1 = 4.0
GAMMA_E1 = 0.08488263631567752
Z_STAR = 1.0
ALPHA = 0.05
T0_PUMP = 50.0
SIGMA_PUMP = 15.0
MD_END = 150.0
DT = 0.005
SAVE_INTERVAL = 100
NN_CUTOFF = 0.65
LATTICE = 36                        # campaign-wide lattice (consistent L=36)

REF_3Q_L18 = CAMPAIGN / "phase_diagram" / "analysis" / "seed3Q_J70400.txt"
REF_ZZ_L18 = CAMPAIGN / "kinetic_barrier" / "zz_relax" / "sample_0" / "spins_T=0.txt"


def me_lambda2(lam_k2: float, all_channels: bool) -> dict[str, float]:
    """Return the four bilinear lambda_{X,2} for a given K-channel value.

    all_channels=True modulates J/K/Gamma/Gammap with the signed Grueneisen
    ratios; False modulates the K channel only.
    """
    if all_channels:
        return {ch: ME_RATIOS[ch] * lam_k2 for ch in ("K", "J", "Gamma", "Gammap")}
    return {"K": lam_k2, "J": 0.0, "Gamma": 0.0, "Gammap": 0.0}


def lam_j7_0(lam_k2: float, j7: float = J7_DEFAULT) -> float:
    """Grueneisen-matched isotropic ring-exchange breathing coupling
    lambda_{J7,0} = (J7/K) lambda_{K,2}  (= +0.001014 at the working point)."""
    return lam_k2 * j7 / K


# --------------------------------------------------------------------------- #
# Lattice geometry + reference states.
# --------------------------------------------------------------------------- #
def _bond_class(xy: np.ndarray, i: int, j: int) -> int:
    vec = xy[j] - xy[i]
    angle = np.degrees(np.arctan2(vec[1], vec[0])) % 180.0
    return int(round((angle - 30.0) / 60.0)) % 3


def _infer_nn_bonds(xy: np.ndarray) -> list[tuple[int, int]]:
    bonds: list[tuple[int, int]] = []
    for i in range(len(xy)):
        delta = xy[i + 1:] - xy[i]
        dist = np.sqrt(np.sum(delta * delta, axis=1))
        for off in np.where(dist < NN_CUTOFF)[0]:
            bonds.append((i, i + 1 + int(off)))
    return bonds


def tile_l18_reference(path: Path, lattice: int) -> np.ndarray:
    """Tile a relaxed 18x18 reference to an LxL lattice (L a multiple of 18)."""
    if lattice % 18 != 0:
        raise ValueError("lattice must be a multiple of 18 to tile the L18 reference")
    reference = np.loadtxt(path).reshape(18, 18, 2, 3)
    reps = lattice // 18
    tiled = np.tile(reference, (reps, reps, 1, 1)).reshape(2 * lattice * lattice, 3)
    return tiled / np.linalg.norm(tiled, axis=1)[:, None]


class LatticeContext:
    """Positions, NN bonds, bond classes and 3Q/ZZ references for an LxL lattice."""

    def __init__(self, lattice: int = LATTICE):
        self.lattice = lattice
        self.n_sites = 2 * lattice * lattice
        self.positions = honeycomb_positions(self.n_sites)
        self.xy = self.positions[:, :2]
        self.x = self.xy[:, 0]
        self.y = self.xy[:, 1]
        self.xc = 0.5 * (float(self.x.min()) + float(self.x.max()))
        self.bonds = _infer_nn_bonds(self.xy)
        self.classes = np.array([_bond_class(self.xy, i, j) for i, j in self.bonds])
        self.bmid = np.array([0.5 * (self.xy[i] + self.xy[j]) for i, j in self.bonds])
        self.q3 = triple_q_state(lattice, lattice)
        self.zz = tile_l18_reference(REF_ZZ_L18, lattice)


def r3_of(spins: np.ndarray, ctx: LatticeContext) -> float:
    """Ratio min/max of the three M-point structure factors (1 = balanced 3Q,
    ->0 = single-M zigzag)."""
    values = []
    for qvec in M_VECTORS:
        phase = ctx.xy[:, 0] * qvec[0] + ctx.xy[:, 1] * qvec[1]
        sq = np.sum(spins * np.exp(1j * phase[:, None]), axis=0)
        values.append(float(np.real(sq @ np.conj(sq))) / ctx.n_sites)
    return min(values) / max(values) if max(values) > 0 else 1.0


def zz_fraction(spins: np.ndarray, ctx: LatticeContext) -> float:
    """Fraction of sites aligned (>0.8) with the relaxed single-domain ZZ reference."""
    return float((np.einsum("ij,ij->i", spins, ctx.zz) > 0.8).mean())


# --------------------------------------------------------------------------- #
# Config builders.  Key sets are the verbatim supersets proven to run in the
# original drivers; only the lattice, working point and coupling vary.
# --------------------------------------------------------------------------- #
def _rel(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def sa_config(*, lattice: int, j7: float, output_dir: Path, initial_spin: Path,
              n_deterministics: int, disorder_file: Path | None = None) -> str:
    """T->0 deterministic relaxation (no drive; quadratic E1 coupling is inert
    at eps=0 so lambda is set to zero here)."""
    disorder = (f"nn_exchange_channel_disorder_config = {_rel(disorder_file)}\n"
                if disorder_file is not None else "")
    return f"""system = NCTO
lattice_size = {lattice},{lattice},1
simulation_mode = SA
output_dir = {_rel(output_dir)}
num_trials = 1
J = {J}
K = {K}
Gamma = {GAMMA}
Gammap = {GAMMAP}
J2_A = {J2_A}
J2_B = {J2_B}
J3 = {J3}
J7 = {j7}
field_strength = 0.0
field_direction = 0,0,1
alpha_gilbert = 0.0
omega_E1 = {OMEGA_E1}
gamma_E1 = {GAMMA_E1}
Z_star = {Z_STAR}
lambda_E1_K_2 = 0.0
pump_amplitude = 0.0
probe_amplitude = 0.0
adiabatic_phonons = false
relax_phonons = false
phonon_only_relax = false
T_start = 1e-9
T_end = 1e-9
annealing_steps = 1
cooling_rate = 0.9
overrelaxation_rate = 0
T_zero = true
n_deterministics = {n_deterministics}
initial_spin_config = {_rel(initial_spin)}
{disorder}"""


def md_config(*, lattice: int, j7: float, output_dir: Path, initial_spin: Path,
              lam2: dict[str, float], lam_j7: float, e0: float,
              pump_polarization: float = 0.0, integrator: str = "rk4",
              disorder_file: Path | None = None) -> str:
    """Driven E1 molecular-dynamics run."""
    disorder = (f"nn_exchange_channel_disorder_config = {_rel(disorder_file)}\n"
                if disorder_file is not None else "")
    return f"""system = NCTO
lattice_size = {lattice},{lattice},1
simulation_mode = MD
output_dir = {_rel(output_dir)}
num_trials = 1
J = {J}
K = {K}
Gamma = {GAMMA}
Gammap = {GAMMAP}
J2_A = {J2_A}
J2_B = {J2_B}
J3 = {J3}
J7 = {j7}
field_strength = 0.0
field_direction = 0,0,1
omega_E1 = {OMEGA_E1}
gamma_E1 = {GAMMA_E1}
lambda_E1_quartic = 0.0
Z_star = {Z_STAR}
lambda_E1_K_2 = {lam2['K']:.12g}
lambda_E1_J_2 = {lam2['J']:.12g}
lambda_E1_Gamma_2 = {lam2['Gamma']:.12g}
lambda_E1_Gammap_2 = {lam2['Gammap']:.12g}
lambda_E1_K_0 = 0.0
lambda_E1_J_0 = 0.0
lambda_E1_Gamma_0 = 0.0
lambda_E1_Gammap_0 = 0.0
lambda_E1_J7_0 = {lam_j7:.12g}
alpha_gilbert = {ALPHA}
langevin_temperature = 0.0
pump_amplitude = {e0:.12g}
pump_frequency = {OMEGA_E1:.12g}
pump_time = {T0_PUMP}
pump_width = {SIGMA_PUMP}
pump_phase = 0.0
pump_polarization = {pump_polarization:.12g}
probe_amplitude = 0.0
md_time_start = 0.0
md_time_end = {MD_END}
md_timestep = {DT}
md_save_interval = {SAVE_INTERVAL}
md_integrator = {integrator}
md_abs_tol = 1e-8
md_rel_tol = 1e-8
initial_spin_config = {_rel(initial_spin)}
{disorder}relax_phonons = false
adiabatic_phonons = false
phonon_only_relax = false
"""


# --------------------------------------------------------------------------- #
# Solver invocation (OMP-pinned to avoid oversubscription) + energy eval.
# --------------------------------------------------------------------------- #
def _solver_env() -> dict:
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = "1"
    return env


def run_solver(cfg: Path) -> bool:
    """Run the solver on a .param file; return True on exit code 0."""
    proc = subprocess.run([str(EXE), str(cfg)], capture_output=True,
                          text=True, cwd=str(ROOT), env=_solver_env())
    if proc.returncode != 0:
        print(f"[solver FAIL] {cfg.stem}: {proc.stderr[-400:]}", flush=True)
    return proc.returncode == 0


def sa_relax(*, lattice: int, j7: float, work_dir: Path, seed_spins: np.ndarray,
             n_deterministics: int, disorder_file: Path | None = None,
             force: bool = False) -> np.ndarray | None:
    """Relax `seed_spins` to T=0 and return the relaxed configuration (cached)."""
    spin_out = work_dir / "sample_0" / "spins_T=0.txt"
    if spin_out.exists() and not force:
        return np.loadtxt(spin_out)
    work_dir.mkdir(parents=True, exist_ok=True)
    seed_path = work_dir / "seed_in.txt"
    np.savetxt(seed_path, seed_spins, fmt="%.10e")
    cfg = work_dir / "relax.param"
    cfg.write_text(sa_config(lattice=lattice, j7=j7, output_dir=work_dir,
                             initial_spin=seed_path, n_deterministics=n_deterministics,
                             disorder_file=disorder_file))
    if not run_solver(cfg) or not spin_out.exists():
        return None
    return np.loadtxt(spin_out)


def energy_config(*, lattice: int, j7: float, output_dir: Path,
                  disorder_file: Path | None = None) -> str:
    """Minimal Hamiltonian config for gneb_field_eval (spins come from the
    manifest, so no initial_spin_config; matches the proven GNEB usage)."""
    disorder = (f"nn_exchange_channel_disorder_config = {_rel(disorder_file)}\n"
                if disorder_file is not None else "")
    return f"""system = NCTO
lattice_size = {lattice},{lattice},1
simulation_mode = SA
output_dir = {_rel(output_dir)}
num_trials = 1
J = {J}
K = {K}
Gamma = {GAMMA}
Gammap = {GAMMAP}
J2_A = {J2_A}
J2_B = {J2_B}
J3 = {J3}
J7 = {j7}
field_strength = 0.0
field_direction = 0,0,1
omega_E1 = {OMEGA_E1}
gamma_E1 = {GAMMA_E1}
lambda_E1_K_2 = 0.0
{disorder}"""


def energy_of(*, lattice: int, j7: float, work_dir: Path, spins: np.ndarray,
              disorder_file: Path | None = None) -> float | None:
    """Total energy (meV) of a spin configuration via gneb_field_eval."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg = work_dir / "energy.param"
    cfg.write_text(energy_config(lattice=lattice, j7=j7, output_dir=work_dir,
                                 disorder_file=disorder_file))
    spin_file = work_dir / "cfg_spins.txt"
    field_file = work_dir / "cfg_H.txt"
    manifest = work_dir / "manifest.txt"
    np.savetxt(spin_file, spins, fmt="%.10e")
    manifest.write_text(f"{spin_file} 0.0 0.0 0 {field_file}\n")
    out_csv = work_dir / "band_e.csv"
    proc = subprocess.run([str(FIELD_EVAL), str(cfg), str(manifest), str(out_csv)],
                          capture_output=True, text=True, cwd=str(ROOT), env=_solver_env())
    if proc.returncode != 0 or not out_csv.exists():
        print(f"[energy FAIL] {work_dir.name}: {proc.stderr[-400:]}", flush=True)
        return None
    rows = np.atleast_2d(np.loadtxt(out_csv, delimiter=",", skiprows=1))
    return float(rows[0, 1] * (2 * lattice * lattice))


# --------------------------------------------------------------------------- #
# Shared parallel runner.  One pool, OMP pinned per subprocess; default width
# is the full core count so studies stay saturated.
# --------------------------------------------------------------------------- #
def default_workers() -> int:
    return os.cpu_count() or 4


def parallel_run(fn, items, workers: int, label: str = "jobs"):
    """Map `fn` over `items` with a thread pool (each call spawns a subprocess).

    Yields (index, result) as each completes; results with value None are
    still yielded so callers can track progress.
    """
    total = len(items)
    print(f"{label}: {total} tasks on {workers} workers", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): item for item in items}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            yield done, future.result()
            if done % 25 == 0 or done == total:
                print(f"  {label}: {done}/{total}", flush=True)
