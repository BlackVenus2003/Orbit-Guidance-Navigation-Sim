#!/usr/bin/env python3
"""
main_gnc.py

Phase 6 integrator for your NavSat project:
 - uses orbit_parser.load_gmat_ephemeris (if available)
 - uses guide_man planners & finite_thrust_burn to simulate a realistic finite burn
 - propagates post-burn coast with guide_man.propagate_two_body
 - runs EKF (orbit_nav.ekf_predict / ekf_update) on the post-burn coast using GPS-like measurements
 - plots results (3D orbits, Δv vs time, EKF position error)

Drop into your NavSat/src/ directory and run:
    python main_gnc.py
"""
import os, sys, math
import numpy as np
import matplotlib.pyplot as plt

# ---- try to import your modules (they should be in same src folder) ----
try:
    from orbit_parser import load_gmat_ephemeris
    print("[OK] orbit_parser.load_gmat_ephemeris available")
except Exception as e:
    load_gmat_ephemeris = None
    print("[WARN] orbit_parser.load_gmat_ephemeris not available:", e)

try:
    import guide_man as gm
    print("[OK] guide_man imported")
except Exception as e:
    gm = None
    print("[ERROR] guide_man import failed:", e); raise

try:
    import planechange as pc
    print("[OK] planechange imported")
except Exception:
    pc = None

try:
    import orbit_nav as on
    print("[OK] orbit_nav imported (EKF funcs available)")
except Exception as e:
    on = None
    print("[ERROR] orbit_nav import failed:", e); raise

# ---- paths ----
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, "..", "data")
EPHEM_FILE = os.path.join(DATA_DIR, "NavSat.e")

# ---- constants from modules (prefer module constants if present) ----
MU = getattr(gm, "MU_EARTH", None) or getattr(pc, "MU", None) or getattr(on, "MU", None) or 398600.4418e9
RE = getattr(gm, "RE", None) or getattr(pc, "Re", None) or getattr(on, "RE", None) or 6371e3

# ---- helper to load initial state ----
def load_initial_state(ephem_path=EPHEM_FILE):
    # 1) try orbit_parser.load_gmat_ephemeris (returns DataFrame)
    if load_gmat_ephemeris is not None:
        try:
            df = load_gmat_ephemeris(ephem_path)
            # df columns: Time,X,Y,Z,VX,VY,VZ in km / km/s per your orbit_parser
            r_km = df.loc[0, ["X","Y","Z"]].to_numpy().astype(float)
            v_km_s = df.loc[0, ["VX","VY","VZ"]].to_numpy().astype(float)
            r0 = r_km * 1000.0
            v0 = v_km_s * 1000.0
            ephem_rs = df[["X","Y","Z"]].to_numpy() * 1000.0
            print("[OK] loaded initial state from orbit_parser (first ephemeris row)")
            return r0, v0, ephem_rs
        except Exception as e:
            print("[WARN] load_gmat_ephemeris failed:", e)

    # 2) try guide_man.parse_stk_ephemeris or planechange.parse_stk_ephemeris or orbit_nav.parse_stk_ephemeris
    for mod in (gm, pc, on):
        if mod is None:
            continue
        if hasattr(mod, "parse_stk_ephemeris"):
            try:
                t, r_hist, v_hist = mod.parse_stk_ephemeris(ephem_path)
                if len(t) > 0:
                    print(f"[OK] loaded initial state via {mod.__name__}.parse_stk_ephemeris")
                    return r_hist[0], v_hist[0], r_hist
            except Exception as e:
                print(f"[WARN] {mod.__name__}.parse_stk_ephemeris failed:", e)

    # 3) fallback synthetic circular LEO (400 km)
    print("[INFO] Falling back to synthetic circular orbit at 400 km")
    r0 = np.array([RE + 400e3, 0.0, 0.0])
    v0 = np.array([0.0, math.sqrt(MU / (RE + 400e3)), 0.0])
    return r0, v0, None

# ---- main integrated workflow ----
def main():
    # 1) Load initial state
    r0, v0, ephem_rs = load_initial_state(EPHEM_FILE)
    print(f"Initial r0 (km): {r0/1000.0}, v0 (km/s): {v0/1000.0}")

    # 2) Plan Hohmann (use guide_man.hohmann_dv which expects radii in meters)
    r_mag = np.linalg.norm(r0)
    target_alt = 450e3            # target altitude (m) for demo
    r_target = RE + target_alt
    dv1, dv2 = gm.hohmann_dv(r_mag, r_target, mu=MU)
    print(f"[GUIDANCE] Hohmann dv1 = {dv1:.3f} m/s, dv2 = {dv2:.3f} m/s, total = {dv1+dv2:.3f} m/s")

    # form dv1 vector in inertial along prograde (velocity direction)
    v_unit = v0 / (np.linalg.norm(v0) + 1e-12)
    dv1_vec = v_unit * dv1

    # 3) Execute finite-thrust burn to deliver dv1 using guide_man.finite_thrust_burn
    print("[EXECUTE] Starting finite-thrust burn to deliver dv1 (this may take a few seconds)...")
    # choose thrust, isp and mass from guide_man defaults or override
    thrust_N = getattr(gm, "THRUST_N", 0.5)
    isp = getattr(gm, "ISP", 300.0)
    mass0 = getattr(gm, "MASS0", 14.0)
    # burn direction in inertial: dv1_vec normalized
    burn_dir_unit = dv1_vec / (np.linalg.norm(dv1_vec) + 1e-12)
    desired_dv_mag = np.linalg.norm(dv1_vec)
    final_burn_state, burn_hist = gm.finite_thrust_burn(r0, v0, burn_dir_unit, desired_dv_mag,
                                                       mass0=mass0, thrust=thrust_N, Isp=isp, dt=getattr(gm,"DT",1.0), mu=getattr(gm,"MU_EARTH",MU))
    print(f"[EXECUTE] Finite burn finished at t={final_burn_state['t']:.1f}s, true Δv achieved = {np.linalg.norm(final_burn_state['v'] - v0):.4f} m/s")

    # 4) propagate post-burn coast using guide_man.propagate_two_body
    sim_dt = 10.0
    sim_time = 1800.0  # 30 min coast for visualization and EKF run
    tspan = (0.0, sim_time)
    ts_post, rs_post, vs_post = gm.propagate_two_body(final_burn_state['r'], final_burn_state['v'], tspan, dt=sim_dt, mu=getattr(gm,"MU_EARTH",MU))

    # 5) run EKF navigation (use orbit_nav functions) on the post-burn truth
    # orbit_nav provides simulate_truth_and_measurements, ekf_predict, ekf_update
    # We'll call simulate_truth_and_measurements with final_burn_state as truth initial
    ekf_dt = getattr(on, "DT", 1.0)
    sim_duration = min(getattr(on, "SIM_DURATION", 1800), sim_time)
    times, truth_states, meas_times, z_meas = on.simulate_truth_and_measurements(final_burn_state['r'], final_burn_state['v'],
                                                                                 dt=ekf_dt, sim_time=sim_duration,
                                                                                 gps_rate=getattr(on, "GPS_RATE", 1.0),
                                                                                 gps_sigma=getattr(on, "GPS_SIGMA", 20.0))
    # EKF init (use orbit_nav default-like init)
    x_est = truth_states[0].copy() + np.hstack((np.random.normal(0,50,3), np.random.normal(0,0.1,3)))
    P = np.diag([100.0**2, 100.0**2, 100.0**2, 1.0**2, 1.0**2, 1.0**2])
    qacc = getattr(on, "PROCESS_ACCEL_NOISE", 1e-6)
    Q = np.zeros((6,6)); Q[3:6,3:6] = qacc * np.eye(3) * ekf_dt
    R = np.eye(3) * (getattr(on, "GPS_SIGMA", 20.0)**2)

    x_hist = np.zeros((len(times),6))
    P_trace = np.zeros(len(times))
    meas_idx = 0

    for k,t in enumerate(times):
        x_pred, P = on.ekf_predict(x_est, P, Q, ekf_dt)
        if meas_idx < len(meas_times) and abs(meas_times[meas_idx] - t) < 1e-6:
            z = z_meas[meas_idx]
            x_est, P, K, y = on.ekf_update(x_pred, P, z, R)
            meas_idx += 1
        else:
            x_est = x_pred
        x_hist[k] = x_est.copy()
        P_trace[k] = np.trace(P)

    # compute position error
    pos_err = np.linalg.norm(truth_states[:,0:3] - x_hist[:,0:3], axis=1)

    # 6) plotting: ephemeris (if loaded), burn history, post-burn coast, EKF error
    fig = plt.figure(figsize=(14,9))
    ax3d = fig.add_subplot(221, projection='3d')
    if ephem_rs is not None:
        ax3d.plot(ephem_rs[:,0]/1000.0, ephem_rs[:,1]/1000.0, ephem_rs[:,2]/1000.0, color='gray', alpha=0.3, label='ephemeris')
    # plot pre-burn ephem point and burn arc
    # pre-burn orbit (a short pre segment from burn_hist if present)
    if burn_hist.get("r") is not None and burn_hist["r"].size > 0:
        ax3d.plot(burn_hist["r"][:,0]/1000.0, burn_hist["r"][:,1]/1000.0, burn_hist["r"][:,2]/1000.0, color='orange', label='during burn')
    # post burn coast
    if rs_post is not None and rs_post.size>0:
        ax3d.plot(rs_post[:,0]/1000.0, rs_post[:,1]/1000.0, rs_post[:,2]/1000.0, color='red', label='post-burn coast')
    ax3d.scatter([r0[0]/1000.0],[r0[1]/1000.0],[r0[2]/1000.0], color='green', s=30, label='burn start (init)')
    ax3d.set_title("Orbit (km) - burn & coast"); ax3d.legend(); ax3d.set_box_aspect([1,1,1])

    # Δv vs time during finite burn
    ax_dv = fig.add_subplot(222)
    if burn_hist.get("t") is not None and burn_hist["t"].size>0:
        if "true_dv" in burn_hist:
            ax_dv.plot(burn_hist["t"], burn_hist["true_dv"], label='true Δv')
        if "measured_dv" in burn_hist:
            ax_dv.plot(burn_hist["t"], burn_hist["measured_dv"], label='measured Δv')
    ax_dv.axhline(desired_dv_mag, color='k', linestyle='--', label='planned dv1')
    ax_dv.set_xlabel("time (s)"); ax_dv.set_ylabel("Δv (m/s)"); ax_dv.set_title("Finite burn Δv"); ax_dv.grid(True); ax_dv.legend()

    # EKF position error
    ax_err = fig.add_subplot(223)
    ax_err.plot(times/60.0, pos_err)
    ax_err.set_xlabel("time (min)"); ax_err.set_ylabel("position error (m)"); ax_err.set_title("EKF Position Error (post-burn)"); ax_err.grid(True)

    # final diagnostics text
    ax_text = fig.add_subplot(224)
    ax_text.axis('off')
    ax_text.text(0.01, 0.95, f"Planned dv1 (m/s): {dv1:.4f}", fontsize=10)
    ax_text.text(0.01, 0.90, f"True dv achieved (m/s): {np.linalg.norm(final_burn_state['v'] - v0):.6f}", fontsize=10)
    ax_text.text(0.01, 0.85, f"Mass before: {mass0:.3f} kg", fontsize=10)
    ax_text.text(0.01, 0.80, f"Mass after burn: {final_burn_state['m']:.3f} kg", fontsize=10)
    ax_text.text(0.01, 0.75, f"EKF final pos err (m): {pos_err[-1]:.3f}", fontsize=10)

    plt.tight_layout()
    plt.show()

    print("=== Phase-6 run complete ===")

if __name__ == "__main__":
    main()
