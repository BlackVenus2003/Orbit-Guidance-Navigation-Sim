#!/usr/bin/env python3
"""
phase4_maneuvers_allinone.py

Phase 4 integrated script:
 - Loads GMAT/STK .e ephemeris (optional) for initial r,v
 - Provides tools to plan and simulate:
     * Hohmann transfer (impulsive)
     * Instantaneous inclination change
     * Combined Hohmann + plane-change (apply combined delta-v vector at chosen point)
     * Finite-thrust burn (simple continuous thrust, mass loss)
 - Visualizes pre/transfer/post orbits in 3D and delta-v diagnostics.

Author: assistant (for your project)
"""

import numpy as np
import math
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import ode
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------
# User config
# ----------------------------
EPHEM_PATH = "../data/NavSat.e"   # set path; if not present set USE_EPHEM=False
USE_EPHEM = True                  # if False script will use user-defined circular orbit
MU_EARTH = 398600.4418e9          # m^3/s^2
RE = 6371e3                       # m

# Default circular orbit altitudes (used if not using ephemeris)
ALT_A = 400e3   # initial altitude (m)
ALT_B = 700e3   # target altitude (m)

# Finite-thrust parameters (for continuous burn simulation)
THRUST_N = 0.5            # N
ISP = 300.0               # s
MASS0 = 14.0              # kg total initial mass
DRY_MASS = 12.0           # kg
DT = 1.0                  # integrator time step for finite thrust (s)

# Plot preferences
SHOW_3D = True
SHOW_2D = True

# ----------------------------
# Utilities
# ----------------------------
def norm(v):
    return np.linalg.norm(v)

def parse_stk_ephemeris(file_path):
    """Parse STK/GMAT .e ephemeris file (extract numeric rows with 7 columns)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    rows = []
    with open(file_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # skip headers
            if s.startswith(("BEGIN","END","ScenarioEpoch","CentralBody","CoordinateSystem",
                             "DistanceUnit","NumberOfEphemerisPoints","EphemerisTimePosVel","WrittenBy")):
                continue
            parts = s.split()
            try:
                nums = [float(x) for x in parts[:7]]
                rows.append(nums)
            except:
                continue
    if len(rows) == 0:
        raise ValueError("No numeric rows found in ephemeris.")
    arr = np.array(rows)
    t = arr[:,0]
    r_km = arr[:,1:4]
    v_km_s = arr[:,4:7]
    return t, r_km*1000.0, v_km_s*1000.0

def coes_from_rv(r, v, mu=MU_EARTH):
    """Return (a, e, i_deg, RAAN_deg, argp_deg, nu_deg) from r (m), v (m/s)."""
    r = np.array(r); v = np.array(v)
    rmag = norm(r); vmag = norm(v)
    h = np.cross(r, v); hmag = norm(h)
    evec = (np.cross(v, h)/mu) - r/rmag
    e = norm(evec)
    energy = vmag**2/2 - mu/rmag
    a = -mu/(2*energy) if abs(energy) > 1e-12 else np.inf
    i = math.degrees(math.acos(np.clip(h[2]/(hmag+1e-16), -1.0, 1.0)))
    # RAAN
    K = np.array([0.0,0.0,1.0])
    nvec = np.cross(K, h); n = norm(nvec)
    if n > 1e-12:
        RAAN = math.degrees(math.acos(np.clip(nvec[0]/n, -1.0, 1.0)))
        if nvec[1] < 0:
            RAAN = 360.0 - RAAN
    else:
        RAAN = 0.0
    # argp
    if n>1e-12 and e>1e-12:
        argp = math.degrees(math.acos(np.clip(np.dot(nvec, evec)/(n*e), -1.0, 1.0)))
        if evec[2] < 0:
            argp = 360.0 - argp
    else:
        argp = 0.0
    # nu
    if e>1e-12:
        nu = math.degrees(math.acos(np.clip(np.dot(evec, r)/(e*rmag), -1.0, 1.0)))
        if np.dot(r,v) < 0:
            nu = 360.0 - nu
    else:
        nu = 0.0
    return a, e, i, RAAN, argp, nu

def propagate_two_body(r0, v0, tspan, dt=10.0, mu=MU_EARTH):
    """Simple RK4 integration of two-body; returns times, rs, vs."""
    def deriv(t, y):
        r = y[0:3]; v = y[3:6]
        rmag = norm(r)
        a = -mu * r / (rmag**3)
        return np.hstack((v, a))
    # use scipy.ode
    solver = ode(deriv).set_integrator('dopri5', atol=1e-9, rtol=1e-9)
    y0 = np.hstack((r0, v0))
    solver.set_initial_value(y0, tspan[0])
    ts = [tspan[0]]; ys = [y0.copy()]
    while solver.successful() and solver.t < tspan[1]-1e-9:
        tnext = min(solver.t + dt, tspan[1])
        solver.integrate(tnext)
        ts.append(solver.t); ys.append(solver.y.copy())
    arr = np.array(ys)
    rs = arr[:,0:3]; vs = arr[:,3:6]
    return np.array(ts), rs, vs

# ----------------------------
# Maneuver planners
# ----------------------------
def hohmann_dv(r1, r2, mu=MU_EARTH):
    """Return Δv1, Δv2 (both magnitudes) for Hohmann from circular radius r1->r2."""
    a_t = 0.5*(r1 + r2)
    v1 = math.sqrt(mu / r1)
    v2 = math.sqrt(mu / r2)
    v_perigee_transfer = math.sqrt(2*mu*r2 / (r1*(r1+r2)))
    v_apogee_transfer = math.sqrt(2*mu*r1 / (r2*(r1+r2)))
    dv1 = abs(v_perigee_transfer - v1)
    dv2 = abs(v2 - v_apogee_transfer)
    return dv1, dv2

def inclination_change_dv(v, delta_i_rad):
    """Instant plane-change magnitude at speed v for inclination change delta_i_rad."""
    # Δv = 2 v sin(Δi/2) (if you rotate velocity vector by Δi)
    return 2.0 * v * abs(math.sin(delta_i_rad/2.0))

def combined_impulsive_dv(v_vec, dv_scaling, rotate_axis_unit=None, delta_i_rad=0.0):
    """
    Example helper: compute vector Δv to achieve both magnitude change (scaling along v_vec)
    and rotate by delta_i_rad about rotate_axis_unit.
    This function returns final dv vector and its magnitude.
    """
    # scale speed by dv_scaling factor (v_final = (1+alpha)*v)
    v0 = np.array(v_vec)
    v0mag = norm(v0)
    if v0mag < 1e-12:
        return np.zeros(3), 0.0
    # first scale (approx radial in velocity direction)
    v_scaled = v0 * (1.0 + dv_scaling)
    # second rotate by delta_i around rotate_axis_unit using Rodrigues
    if rotate_axis_unit is None or abs(delta_i_rad) < 1e-12:
        v_final = v_scaled
    else:
        k = rotate_axis_unit / (norm(rotate_axis_unit)+1e-16)
        v = v_scaled
        cosA = math.cos(delta_i_rad); sinA = math.sin(delta_i_rad)
        v_final = v * cosA + np.cross(k, v) * sinA + k * (np.dot(k, v)) * (1 - cosA)
    dv_vec = v_final - v0
    return dv_vec, norm(dv_vec)

# ----------------------------
# Finite thrust (continuous burn) executor
# ----------------------------
def finite_thrust_burn(r0, v0, burn_dir_unit, desired_dv_mag, mass0=MASS0, thrust=THRUST_N, Isp=ISP, dt=DT, mu=MU_EARTH):
    """
    Simulate finite-thrust burn: apply constant thrust in inertial burn_dir_unit (assumed inertial constant),
    integrate until delivered delta-v equals desired_dv_mag or propellant exhausted.
    Returns final (r,v,mass), time history dict (t, r, v, mass, true_dv)
    Note: crude model (no attitude dynamics), thrust vector fixed in inertial frame.
    """
    burn_dir_unit = np.array(burn_dir_unit) / (norm(burn_dir_unit)+1e-16)
    state = np.zeros(7)
    state[0:3] = r0.copy()
    state[3:6] = v0.copy()
    state[6] = mass0
    t = 0.0
    history = {"t":[],"r":[],"v":[],"m":[],"true_dv":[]}
    v_init = v0.copy()
    mdot = -thrust / (Isp * 9.80665)  # kg/s negative
    # integrate with simple RK4 for coupled motion and mass change
    def deriv(s):
        r = s[0:3]; v = s[3:6]; m = s[6]
        rmag = norm(r); a_grav = -mu * r / (rmag**3)
        if m <= DRY_MASS or thrust <= 0.0:
            a_thrust = np.zeros(3)
        else:
            a_thrust = (thrust / m) * burn_dir_unit
        ds = np.zeros(7)
        ds[0:3] = v
        ds[3:6] = a_grav + a_thrust
        ds[6] = mdot
        return ds
    max_steps = int(7200.0/dt)
    for step in range(max_steps):
        # record
        history["t"].append(t)
        history["r"].append(state[0:3].copy())
        history["v"].append(state[3:6].copy())
        history["m"].append(state[6])
        true_dv = norm(state[3:6] - v_init)
        history["true_dv"].append(true_dv)
        if true_dv >= desired_dv_mag - 1e-6:
            break
        # RK4 step
        k1 = deriv(state)
        k2 = deriv(state + 0.5*dt*k1)
        k3 = deriv(state + 0.5*dt*k2)
        k4 = deriv(state + dt*k3)
        state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        # clamp mass
        if state[6] < DRY_MASS:
            state[6] = DRY_MASS
            # stop if no propellant
            break
        t += dt
    final = {"r": state[0:3], "v": state[3:6], "m": state[6], "t": t}
    for k in history:
        history[k] = np.array(history[k])
    return final, history

# ----------------------------
# Visualization helpers
# ----------------------------
def plot_3d_orbits(ephem_rs=None, pre_rs=None, transfer_rs=None, post_rs=None, burn_point=None, title="Orbits"):
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(111, projection='3d')
    if ephem_rs is not None:
        ax.plot(ephem_rs[:,0]/1000.0, ephem_rs[:,1]/1000.0, ephem_rs[:,2]/1000.0, color='gray', alpha=0.3, label='Ephemeris')
    if pre_rs is not None:
        ax.plot(pre_rs[:,0]/1000.0, pre_rs[:,1]/1000.0, pre_rs[:,2]/1000.0, color='green', label='Pre-burn')
    if transfer_rs is not None:
        ax.plot(transfer_rs[:,0]/1000.0, transfer_rs[:,1]/1000.0, transfer_rs[:,2]/1000.0, color='orange', label='Transfer')
    if post_rs is not None:
        ax.plot(post_rs[:,0]/1000.0, post_rs[:,1]/1000.0, post_rs[:,2]/1000.0, color='red', label='Post')
    if burn_point is not None:
        ax.scatter([burn_point[0]/1000.0], [burn_point[1]/1000.0], [burn_point[2]/1000.0], color='blue', s=80, label='Burn point')
    # Earth
    u,v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = (RE/1000.0)*np.cos(u)*np.sin(v)
    y = (RE/1000.0)*np.sin(u)*np.sin(v)
    z = (RE/1000.0)*np.cos(v)
    ax.plot_surface(x,y,z, color='lightblue', alpha=0.5)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Z (km)')
    ax.set_title(title)
    ax.legend()
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.show()

def plot_time_series(t, series_list, labels, title, ylabel):
    plt.figure(figsize=(8,4))
    for s,l in zip(series_list, labels):
        plt.plot(t, s, label=l)
    plt.xlabel('Time (s)'); plt.ylabel(ylabel); plt.title(title); plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------
# Phase-4 integrated runner
# ----------------------------
def main():
    # initial state
    if USE_EPHEM and os.path.exists(EPHEM_PATH):
        try:
            t_e, r_e, v_e = parse_stk_ephemeris(EPHEM_PATH)
            print(f"Loaded ephemeris points: {len(t_e)}")
            # use first ephemeris row as initial state
            r0 = r_e[0]; v0 = v_e[0]
            ephem_for_plot = r_e
        except Exception as e:
            print("Failed to parse ephemeris:", e)
            print("Falling back to circular altitudes.")
            r0 = np.array([RE + ALT_A, 0.0, 0.0])
            v0 = np.array([0.0, math.sqrt(MU_EARTH/(RE+ALT_A)), 0.0])
            ephem_for_plot = None
    else:
        # initial circular orbit in x-y plane
        r0 = np.array([RE + ALT_A, 0.0, 0.0])
        v0 = np.array([0.0, math.sqrt(MU_EARTH/(RE+ALT_A)), 0.0])
        ephem_for_plot = None
    print("Initial r0 (km):", r0/1000.0, "v0 (km/s):", v0/1000.0)
    a0, e0, i0, RAAN0, argp0, nu0 = coes_from_rv(r0, v0)
    print(f"Initial orbit a={a0/1000.0:.3f} km, e={e0:.6f}, i={i0:.3f} deg")

    # ------------------
    # 1) Hohmann transfer
    # ------------------
    r1 = norm(r0)
    r2 = RE + ALT_B
    dv1, dv2 = hohmann_dv(r1, r2)
    print(f"Hohmann Δv1 = {dv1:.3f} m/s, Δv2 = {dv2:.3f} m/s, total = {dv1+dv2:.3f} m/s")

    # Build transfer orbit: apply dv1 at perigee (r0 direction)
    v_dir = v0 / (norm(v0)+1e-16)
    # impulsive burn 1 velocity
    v_after1 = v0 + v_dir * dv1
    # propagate transfer from burn time to time at apogee (transfer half period)
    a_transfer = 0.5*(r1 + r2)
    T_transfer = math.pi * math.sqrt(a_transfer**3 / MU_EARTH)  # half period (sec)
    # propagate pre-burn short for plot
    tspan_pre = (0.0, 0.2*3600.0)
    t_pre, rs_pre, vs_pre = propagate_two_body(r0, v0, tspan_pre, dt=30.0)
    # propagate transfer arc from burn epoch to burn + T_transfer
    t_t, rs_t, vs_t = propagate_two_body(r0, v_after1, (0.0, T_transfer), dt=30.0)
    # apply second impulsive burn at apogee: velocity at apogee is last vs_t row
    if vs_t.shape[0] > 0:
        v_at_apogee = vs_t[-1]
        r_at_apogee = rs_t[-1]
        # apply dv2 in direction of velocity at apogee
        v_dir_ap = v_at_apogee / (norm(v_at_apogee)+1e-16)
        v_after2 = v_at_apogee + v_dir_ap * dv2
        # propagate final orbit for 1 orbit after apogee
        tspan_post = (0.0, 3600.0)
        t_post, rs_post, vs_post = propagate_two_body(r_at_apogee, v_after2, tspan_post, dt=60.0)
    else:
        rs_t = None; rs_post = None; r_at_apogee = None

    # Plot Hohmann result
    if SHOW_3D:
        plot_3d_orbits(ephem_for_plot, pre_rs=rs_pre, transfer_rs=rs_t, post_rs=rs_post, burn_point=r0,
                       title="Hohmann Transfer (impulsive)")

    # ------------------
    # 2) Inclination change (instant)
    # ------------------
    # choose burn location as first ascending node of ephemeris if available; otherwise at r0
    burn_r = r0.copy()
    burn_v = v0.copy()
    # desired inclination change
    target_inc_deg = i0 + 5.0  # add 5 deg for demo
    delta_i_rad = math.radians(target_inc_deg - i0)
    v_speed = norm(burn_v)
    dv_plane = inclination_change_dv(v_speed, delta_i_rad)
    print(f"Inclination change Δi = {math.degrees(delta_i_rad):.3f} deg -> Δv ≈ {dv_plane:.3f} m/s (impulsive)")

    # build vector delta-v by rotating velocity vector about ascending node line (choose x-axis as rotate axis for demo)
    # For accurate plane-change vector, rotate v about node line (here approximate rotate about line perpendicular to v and z)
    k = np.cross(burn_v, np.array([0.0,0.0,1.0]))
    if norm(k) < 1e-8:
        k = np.array([1.0, 0.0, 0.0])
    k = k / norm(k)
    # rotate v by delta_i_rad using Rodrigues to get new v
    v = burn_v
    cosA = math.cos(delta_i_rad); sinA = math.sin(delta_i_rad)
    v_rot = v*cosA + np.cross(k, v)*sinA + k*(np.dot(k,v))*(1-cosA)
    dv_vec_plane = v_rot - v
    # propagate small afterburn orbit for plot
    t_plane, rs_plane, vs_plane = propagate_two_body(burn_r, burn_v + dv_vec_plane, (0.0, 1800.0), dt=30.0)
    if SHOW_3D:
        plot_3d_orbits(ephem_for_plot, pre_rs=rs_pre, transfer_rs=None, post_rs=rs_plane, burn_point=burn_r,
                       title=f"Inclination change by {math.degrees(delta_i_rad):.3f} deg (impulsive)")

    # ------------------
    # 3) Combined Hohmann + plane change (apply combined dv at perigee)
    # ------------------
    # at perigee (r0) combine dv1 vector (prograde) and plane rotation about some axis (k)
    dv1_vec = v_dir * dv1
    # rotate dv1_vec by delta_i_rad about k? Instead compute dv that moves v0 -> desired v: scale+rotate
    v_target_combined = v_after1.copy()
    # rotate v_target_combined by delta_i_rad around k to add plane change
    v_combined_rot = v_target_combined*cosA + np.cross(k, v_target_combined)*sinA + k*(np.dot(k,v_target_combined))*(1-cosA)
    dv_combined = v_combined_rot - v0
    dv_combined_mag = norm(dv_combined)
    print(f"Combined impulsive Δv magnitude at burn point: {dv_combined_mag:.3f} m/s (vs separate dv sum {dv1+dv_plane:.3f})")

    # simulate applying dv_combined at t=0 and integrate transfer
    t_comb, rs_comb, vs_comb = propagate_two_body(r0, v0 + dv_combined, (0.0, T_transfer), dt=30.0)
    if SHOW_3D:
        plot_3d_orbits(ephem_for_plot, pre_rs=rs_pre, transfer_rs=rs_comb, post_rs=None, burn_point=r0,
                       title="Combined Hohmann + Plane-change (impulsive)")

    # ------------------
    # 4) Finite-thrust execution for the Hohmann first impulse (demo)
    # ------------------
    # aim to deliver dv1 magnitude along prograde direction using finite thrust
    desired_dv = dv1
    burn_dir = v_dir  # inertial direction
    print(f"Simulating finite-thrust to deliver Δv ≈ {desired_dv:.3f} m/s along prograde direction (thrust={THRUST_N} N)")
    final_f, hist_f = finite_thrust_burn(r0, v0, burn_dir, desired_dv, mass0=MASS0, thrust=THRUST_N, Isp=ISP, dt=DT)
    print("Finite burn ended at t =", final_f["t"], "s, achieved true dv =", norm(final_f["v"] - v0), "m/s, mass left =", final_f["m"], "kg")

    # propagate a bit after finite burn
    t_fin_post, rs_fin_post, vs_fin_post = propagate_two_body(final_f["r"], final_f["v"], (0.0, 3600.0), dt=60.0)
    if SHOW_3D:
        plot_3d_orbits(ephem_for_plot, pre_rs=rs_pre, transfer_rs=np.array(hist_f["r"]), post_rs=rs_fin_post, burn_point=r0,
                       title="Finite-thrust burn (prograde) and coast")

    # plot Δv vs time for finite burn
    if hist_f["t"].size > 0:
        plot_time_series(hist_f["t"], [hist_f["true_dv"]], ['true Δv'], 'Finite burn Δv vs time', 'Δv (m/s)')

    # Done
    print("Phase 4 complete. You can adjust ALT_A, ALT_B, PLANNED delta-vs, or finite-thrust params at the top and re-run.")

if __name__ == "__main__":
    main()
