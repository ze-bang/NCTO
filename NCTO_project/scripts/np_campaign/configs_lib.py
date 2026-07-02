"""Real-mode config templates for the campaign.

Each helper writes a complete .param file for a specific simulation
mode supported by the `spin_solver` binary, rather than going through
the pump-probe base template.  These wrap the modes that the campaign
rewrite needs:

    write_gneb_strain_config        -> simulation_mode = kinetic_barrier
                                       (Bessarab-Uzdin-Jonsson GNEB)
    write_pt_strain_config          -> simulation_mode = PT
                                       (Hukushima-Nemoto parallel tempering)
    write_langevin_pump_config      -> simulation_mode = pump_probe with T>0
    write_langevin_relax_config     -> simulation_mode = MD with T>0
                                       (post-pump droplet decay)
    write_2dcs_config               -> simulation_mode = 2dcs

All paths returned are absolute.  The functions follow the field
conventions in the existing example_configs/NCTO and gneb_ensemble
files verbatim.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def _w(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


# --------------------------------------------------------------------- GNEB
def write_gneb_strain_config(
    path: Path,
    output_dir: str,
    *,
    initial_state_file: str,
    final_state_file: str,
    eps_Eg_magnitude: float = 0.0,
    eps_Eg_direction: float = 0.0,
    J: float = -1.0, K: float = -6.0,
    Gamma: float = 8.0, Gammap: float = -3.5,
    J7: float = -0.010407,
    lambda_Eg: float = 0.10,
    disorder_strength: float = 0.0,
    disorder_seed: int = 0,
    lattice_size: tuple = (6, 6, 1),
    n_images: int = 32,
    max_iterations: int = 50000,
    force_tolerance: float = 1e-4,
    climbing: bool = True,
    num_trials: int = 1,
) -> Path:
    """GNEB-with-fixed-strain run on NCTO_STRAIN.

    The fixed strain plays the role of the time-averaged phonon
    distortion: epsilon_xx = m cos(2 theta), epsilon_yy = -m cos(2 theta),
    epsilon_xy = m sin(2 theta), where m = eps_Eg_magnitude.
    This is the quasi-static (BO) proxy for a driven E1 phonon at
    fixed amplitude |Q|^2 = m^2 and polarisation theta.
    """
    Lx, Ly, Lz = lattice_size
    body = f"""# Auto-generated GNEB (kinetic_barrier) config — campaign C2
system = NCTO_STRAIN
lattice_size = {Lx},{Ly},{Lz}
simulation_mode = kinetic_barrier
num_trials = {num_trials}
output_dir = {output_dir}

J = {J}
K = {K}
Gamma = {Gamma}
Gammap = {Gammap}
J2_A = 0.0
J2_B = 0.0
J3 = 0.0
J7 = {J7}
gamma_J7 = 0.0
field_strength = 0.0
field_direction = 0,0,1

disorder_strength = {disorder_strength}
disorder_seed = {disorder_seed}
dilution_fraction = 0.0

lambda_Eg = {lambda_Eg}
lambda_A1g = 0.0
C11 = 1.0
C12 = 0.3
C44 = 0.35
kappa_Eg = 0.0
kappa_A1g = 0.0
strain_mass = 1.0
gamma_A1g = 0.0
gamma_Eg = 0.0

# Quasi-static externally applied strain proxy for driven phonon.
gneb_external_eps_xx = {eps_Eg_magnitude * _cos2(eps_Eg_direction)}
gneb_external_eps_yy = {-eps_Eg_magnitude * _cos2(eps_Eg_direction)}
gneb_external_eps_xy = {eps_Eg_magnitude * _sin2(eps_Eg_direction)}

gneb_n_images           = {n_images}
gneb_spring_constant    = 1.0
gneb_force_tolerance    = {force_tolerance}
gneb_max_iterations     = {max_iterations}
gneb_use_climbing_image = {"true" if climbing else "false"}
gneb_fixed_strain       = true
gneb_dynamic_strain     = false
gneb_strain_sweep       = false
gneb_save_path_evolution = false

gneb_polish_endpoints  = true
gneb_polish_max_iter   = 5000
gneb_polish_force_tol  = 1e-5

gneb_initial_state_file = {initial_state_file}
gneb_final_state_file   = {final_state_file}
gneb_pin_Eg2_zero = false
"""
    return _w(path, body)


# --------------------------------------------------------------------- PT
def write_pt_strain_config(
    path: Path, output_dir: str, *,
    T_list: list,
    n_anneal: int = 20000, n_measure: int = 20000,
    overrelaxation_rate: int = 5,
    initial_state_file: Optional[str] = None,
    lattice_size: tuple = (6, 6, 1),
    J: float = -1.0, K: float = -6.0,
    Gamma: float = 8.0, Gammap: float = -3.5,
    J7: float = -0.010407,
    lambda_Eg: float = 0.10,
    disorder_strength: float = 0.0, disorder_seed: int = 0,
    pt_ranks_per_point: int = 1,
) -> Path:
    """Parallel tempering on NCTO_STRAIN at the prescribed temperature grid.

    Returns the path to the new .param file.  Caller is responsible
    for running with MPI: `mpirun -n len(T_list) spin_solver cfg`.
    """
    Lx, Ly, Lz = lattice_size
    T_str = ",".join(f"{t:.6g}" for t in T_list)
    seed_block = (f"initial_spin_config = {initial_state_file}\n"
                  if initial_state_file else "")
    body = f"""# Auto-generated parallel tempering config — campaign C3/C4
system = NCTO_STRAIN
lattice_size = {Lx},{Ly},{Lz}
simulation_mode = PT
output_dir = {output_dir}
num_trials = 1

J = {J}
K = {K}
Gamma = {Gamma}
Gammap = {Gammap}
J7 = {J7}
field_strength = 0.0
field_direction = 0,0,1
lambda_Eg = {lambda_Eg}
lambda_A1g = 0.0
C11 = 1.0
C12 = 0.3
C44 = 0.35
strain_mass = 1.0

disorder_strength = {disorder_strength}
disorder_seed = {disorder_seed}

pt_temperatures = {T_str}
pt_n_anneal = {n_anneal}
pt_n_measure = {n_measure}
pt_overrelaxation_rate = {overrelaxation_rate}
pt_ranks_per_point = {pt_ranks_per_point}
{seed_block}"""
    return _w(path, body)


# --------------------------------------------------------------------- Langevin pump
def write_langevin_pump_config(
    path: Path, output_dir: str, *,
    seed_file: str,
    E0: float, theta: float,
    T: float = 0.0,
    J: float = -1.0, K: float = -6.0,
    Gamma: float = 8.0, Gammap: float = -3.5,
    J3: float = 0.9,
    J7: float = -0.0026,
    J2_A: float = 0.0,
    J2_B: float = 0.0,
    omega_E1: float = 4.0,
    # Calibrated via effective-frequency matching (l18_effomega4_cycle15).
    gamma_E1: float = 0.0848826363157,
    lambda_E1_quartic: float = 0.01,
    lambda_K_2: float = 0.035,
    pump_freq: float = 4.0,
    # Calibrated 15-cycle pulse: sigma=15 t.u., peak at t=50.
    pump_t0: float = 50.0, pump_sigma: float = 15.0,
    lattice_size: tuple = (24, 24, 1),
    alpha_G: float = 0.05,
    t_start: float = -4.0, t_end: float = 180.0, dt: float = 0.005,
    phonon_only_relax: bool = False,
) -> Path:
    Lx, Ly, Lz = lattice_size
    integrator = "dopri5" if T <= 0.0 else "stochastic_heun"
    body = f"""# Auto-generated finite-T pump-probe (sLLG) — campaign C5
system = NCTO
simulation_mode = pump_probe
lattice_size = {Lx},{Ly},{Lz}
output_dir = {output_dir}
num_trials = 1

J = {J}
K = {K}
Gamma = {Gamma}
Gammap = {Gammap}
J2_A = {J2_A}
J2_B = {J2_B}
J3 = {J3}
J7 = {J7}
field_strength = 0.0
field_direction = 0,0,1
alpha_gilbert = {alpha_G}
langevin_temperature = {T}

omega_E1 = {omega_E1}
gamma_E1 = {gamma_E1}
lambda_E1_quartic = {lambda_E1_quartic}
Z_star = 1.0
lambda_E1_K_2 = {lambda_K_2}

pump_amplitude    = {E0}
pump_frequency    = {pump_freq}
pump_time         = {pump_t0}
pump_width        = {pump_sigma}
pump_phase        = 0.0
pump_polarization = {theta}
probe_amplitude   = 0.0

md_time_start = {t_start}
md_time_end   = {t_end}
md_timestep   = {dt}
md_save_interval = 10
md_integrator = {integrator}
md_abs_tol = 1e-8
md_rel_tol = 1e-8

initial_spin_config = {seed_file}
relax_phonons = true
adiabatic_phonons = false
phonon_only_relax = {"true" if phonon_only_relax else "false"}

T_start = 0.0
T_end = 0.0
T_zero = true
annealing_steps = 0
"""
    return _w(path, body)


# --------------------------------------------------------------------- Langevin relax (no drive)
def write_langevin_relax_config(
    path: Path, output_dir: str, *,
    seed_file: str,
    T: float, t_max: float,
    lattice_size: tuple = (24, 24, 1),
    alpha_G: float = 0.05,
    disorder_strength: float = 0.0, disorder_seed: int = 0,
) -> Path:
    """Pure MD relaxation under Langevin thermostat, no pump pulse.
    Used by C6 for prepared-droplet decay measurement."""
    Lx, Ly, Lz = lattice_size
    integrator = "stochastic_heun" if T > 0 else "rkf54"
    # NCTO_STRAIN to use disorder hooks if requested.
    system = "NCTO_STRAIN" if disorder_strength > 0 else "NCTO"
    body = f"""# Auto-generated droplet relaxation (Langevin) — campaign C6
system = {system}
simulation_mode = MD
lattice_size = {Lx},{Ly},{Lz}
output_dir = {output_dir}
num_trials = 1

J = -1.0
K = -6.0
Gamma = 8.0
Gammap = -3.5
J7 = -0.010407
field_strength = 0.0
field_direction = 0,0,1
alpha_gilbert = {alpha_G}
langevin_temperature = {T}
lambda_Eg = 0.10

disorder_strength = {disorder_strength}
disorder_seed = {disorder_seed}

pump_amplitude = 0.0
probe_amplitude = 0.0

md_time_start = 0.0
md_time_end   = {t_max}
md_timestep   = 0.01
md_save_interval = 10
md_integrator = {integrator}

initial_spin_config = {seed_file}
"""
    return _w(path, body)


# --------------------------------------------------------------------- 2DCS
def write_2dcs_config(
    path: Path, output_dir: str, *,
    seed_file: str,
    tau_start: float = -50.0, tau_end: float = 50.0, tau_step: float = 0.5,
    omega_E1: float = 4.0,
    E0: float = 1.0,
    lattice_size: tuple = (12, 12, 1),
) -> Path:
    Lx, Ly, Lz = lattice_size
    body = f"""# Auto-generated 2DCS config — campaign C7
system = NCTO
simulation_mode = 2dcs
lattice_size = {Lx},{Ly},{Lz}
output_dir = {output_dir}
num_trials = 1

J = -1.0
K = -6.0
Gamma = 8.0
Gammap = -3.5
J7 = -0.10
field_strength = 0.0
field_direction = 0,0,1
alpha_gilbert = 0.05
langevin_temperature = 0.0

omega_E1 = {omega_E1}
gamma_E1 = 0.2543
lambda_E1_quartic = 0.0
Z_star = 1.0
lambda_E1_K_2 = 0.04

pump_amplitude    = {E0}
pump_frequency    = {omega_E1}
pump_time         = 0.0
pump_width        = 2.0
pump_phase        = 0.0
pump_polarization = 0.0
probe_amplitude   = {E0}
probe_frequency   = {omega_E1}
probe_width       = 2.0

tau_start = {tau_start}
tau_end   = {tau_end}
tau_step  = {tau_step}

md_time_start = -10.0
md_time_end   = {tau_end + 60.0}
md_timestep   = 0.01
md_save_interval = 1
md_integrator = rkf54

initial_spin_config = {seed_file}
"""
    return _w(path, body)


# ---- helpers
import math as _math


def _cos2(theta_rad: float) -> float:
    return _math.cos(2.0 * theta_rad)


def _sin2(theta_rad: float) -> float:
    return _math.sin(2.0 * theta_rad)
