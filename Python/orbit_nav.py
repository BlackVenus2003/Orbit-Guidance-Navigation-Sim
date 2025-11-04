#!/usr/bin/env python3
"""
ekf_orbit_nav.py

Simple Extended Kalman Filter for orbit state estimation (r,v) using:
 - process model: 2-body dynamics (r'', = -mu*r/|r|^3)
 - measurements: noisy GPS position (r_meas)
 - discrete-time EKF with RK4 prediction step and analytic Jacobian

Inputs:
 - Uses ../data/NavSat.e ephemeris (first line) as truth initial state if available,
   else falls back to circular orbit at 400 km altitude.

Outputs:
 - Plots: 3D truth vs estimate, and position error vs time.
"""

import os
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import ode

# Config
EPHEM_PATH = "../data/NavSat.e"
MU = 398600.4418e9  # m^3/s^2
RE = 6371e3
DT = 1.0            # filter timestep (s)
SIM_DURATION = 1800 # seconds to simulate (e.g., 30 minutes)
GPS_RATE = 1.0      # Hz (GPS measurement interval)
GPS_SIGMA = 20.0    # meters (1-sigma position noise)
PROCESS_ACCEL_NOISE = 1e-6  # (m/s^2)^2 spectral density (tunable)

np.set_printoptions(suppress=True, precision=6)

# -------------------------
# Utilities & ephem parser
# -------------------------
def norm(v):
    return np.linalg.norm(v)

def parse_stk_ephemeris(file_path):
    if not os.path.exists(file_path):
        return None
    rows = []
    with open(file_path,'r') as f:
        for line in f:
            s = line.strip()
            if not s: continue
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
        return None
    arr = np.array(rows)
    t = arr[:,0]
    r = arr[:,1:4]*1000.0
    v = arr[:,4:7]*1000.0
    return t, r, v

# -------------------------
# Dynamics & Jacobian
# -------------------------
def two_body_acc(r, mu=MU):
    rnorm = norm(r)
    return -mu * r / (rnorm**3)

def f_state(x):
    """Continuous dynamics f(x) for state x=[r; v]"""
    r = x[0:3]; v = x[3:6]
    a = two_body_acc(r)
    dx = np.zeros(6)
    dx[0:3] = v
    dx[3:6] = a
    return dx

def jacobian_f(x):
    """Analytic Jacobian ∂f/∂x for state x=[r; v]; returns 6x6 matrix."""
    r = x[0:3]; v = x[3:6]
    rmag = norm(r)
    I3 = np.eye(3)
    # partial of acceleration wrt r: d(-mu r / r^3)/dr
    mu = MU
    r_rT = np.outer(r, r)
    dadr = -mu * ( (np.eye(3) * (1.0 / (rmag**3))) - 3.0 * r_rT / (rmag**5) )
    A = np.zeros((6,6))
    A[0:3, 3:6] = I3
    A[3:6, 0:3] = dadr
    return A

# -------------------------
# RK4 integrator for state propagation
# -------------------------
def rk4_step(x, dt):
    k1 = f_state(x)
    k2 = f_state(x + 0.5*dt*k1)
    k3 = f_state(x + 0.5*dt*k2)
    k4 = f_state(x + dt*k3)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# -------------------------
# Discrete EKF functions
# -------------------------
def ekf_predict(x, P, Q, dt):
    """EKF predict: propagate state (RK4) and covariance (linearized)."""
    # propagate state nonlinearly
    x_pred = rk4_step(x, dt)
    # discrete-time linearization: Phi = expm(A*dt) ≈ I + A*dt (small dt) or use simple Taylor
    # We'll approximate Phi ≈ I + A*dt where A = jacobian_f(x) evaluated at current x
    A = jacobian_f(x)
    Phi = np.eye(6) + A*dt  # first-order approx
    P_pred = Phi @ P @ Phi.T + Q
    return x_pred, P_pred

def ekf_update(x_pred, P_pred, z, R):
    """EKF update for measurement z = H x + noise, where H = [I3 0]."""
    H = np.zeros((3,6))
    H[0:3, 0:3] = np.eye(3)
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_upd = x_pred + K @ y
    P_upd = (np.eye(6) - K @ H) @ P_pred
    return x_upd, P_upd, K, y

# -------------------------
# Simulation / measurement generation
# -------------------------
def simulate_truth_and_measurements(r0, v0, dt=DT, sim_time=SIM_DURATION, gps_rate=GPS_RATE, gps_sigma=GPS_SIGMA):
    steps = int(sim_time / dt) + 1
    times = np.arange(0, steps*dt, dt)
    truth_states = np.zeros((len(times), 6))
    truth_states[0,0:3] = r0
    truth_states[0,3:6] = v0
    # propagate truth with RK4
    for k in range(1, len(times)):
        truth_states[k] = rk4_step(truth_states[k-1], dt)
    # generate GPS measurements at gps_rate
    meas_interval = int(round(1.0 / gps_rate / dt))
    meas_times = []
    z_list = []
    for k,t in enumerate(times):
        if k % max(1, meas_interval) == 0:
            r_true = truth_states[k,0:3]
            z = r_true + np.random.normal(0, gps_sigma, 3)
            meas_times.append(t)
            z_list.append(z)
    return times, truth_states, np.array(meas_times), np.array(z_list)

# -------------------------
# Main EKF run
# -------------------------
def main():
    # load ephemeris truth initial state if available
    ep = parse_stk_ephemeris(EPHEM_PATH)
    if ep is not None:
        t_e, r_e, v_e = ep
        print(f"Loaded ephemeris with {len(t_e)} points. Using first row as truth init.")
        r0 = r_e[0]; v0 = v_e[0]
    else:
        print("No ephemeris file found or parse failed - falling back to circular 400 km initial orbit.")
        r0 = np.array([RE + 400e3, 0.0, 0.0])
        v0 = np.array([0.0, math.sqrt(MU/(RE + 400e3)), 0.0])

    # simulate truth and GPS measurements
    times, truth_states, meas_times, z_meas = simulate_truth_and_measurements(r0, v0, dt=DT, sim_time=SIM_DURATION, gps_rate=GPS_RATE, gps_sigma=GPS_SIGMA)
    print(f"Simulated {len(times)} steps, {len(meas_times)} GPS measurements (sigma={GPS_SIGMA} m).")

    # EKF init
    x_est = truth_states[0].copy() + np.hstack((np.random.normal(0,50,3), np.random.normal(0,0.1,3)))  # initial guess noisy
    P = np.diag([100.0**2, 100.0**2, 100.0**2, 1.0**2, 1.0**2, 1.0**2])  # initial covariance
    # process noise Q: approximate discrete process noise from acceleration noise
    qacc = PROCESS_ACCEL_NOISE
    # continuous-time process noise for acceleration -> state Qc = [[0,0],[0,qacc*I]] ; discretize approx: Q = G Qc G^T dt
    # use simple block Q
    Q = np.zeros((6,6))
    Q[3:6,3:6] = qacc * np.eye(3) * DT  # rough approximation
    R = np.eye(3) * (GPS_SIGMA**2)

    # storage
    x_hist = np.zeros((len(times), 6))
    P_trace = np.zeros(len(times))
    meas_idx = 0

    # run filter
    for k,t in enumerate(times):
        # predict
        x_pred, P = ekf_predict(x_est, P, Q, DT)
        # if measurement available at this time, apply update
        if meas_idx < len(meas_times) and abs(meas_times[meas_idx] - t) < 1e-6:
            z = z_meas[meas_idx]
            x_est, P, K, y = ekf_update(x_pred, P, z, R)
            meas_idx += 1
        else:
            x_est = x_pred
        x_hist[k] = x_est.copy()
        P_trace[k] = np.trace(P)

    # compute position error over time
    pos_err = np.linalg.norm(truth_states[:,0:3] - x_hist[:,0:3], axis=1)

    # prints
    print("Final position error (m):", pos_err[-1])
    print("Final velocity error (m/s):", np.linalg.norm(truth_states[-1,3:6] - x_hist[-1,3:6]))

    # plots
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(211)
    ax.plot(times/60.0, pos_err)
    ax.set_xlabel('Time (min)'); ax.set_ylabel('Position error (m)'); ax.grid(True)
    ax.set_title('EKF Position Error')

    ax2 = fig.add_subplot(212, projection='3d')
    ax2.plot(truth_states[:,0]/1000.0, truth_states[:,1]/1000.0, truth_states[:,2]/1000.0, label='Truth')
    ax2.plot(x_hist[:,0]/1000.0, x_hist[:,1]/1000.0, x_hist[:,2]/1000.0, label='EKF estimate')
    ax2.set_xlabel('X (km)'); ax2.set_ylabel('Y (km)'); ax2.set_zlabel('Z (km)')
    ax2.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
