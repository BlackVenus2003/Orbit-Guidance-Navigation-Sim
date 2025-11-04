#!/usr/bin/env python3
"""
Fixed: Project2 - Multi-thruster + reaction-wheel attitude control (bug fixes)

Main fixes:
 - Avoid immediate burn termination at t=0 by requiring a small minimum burn time
 - Ensure planned Δv fallback if zero
 - Add debug prints for ephemeris and history shapes
 - Robust plotting even when history arrays are small
"""

import numpy as np
import math
import os
import matplotlib.pyplot as plt

# ------------------------
# Config / constants
# ------------------------
EPHEM_PATH = "../data/NavSat.e"  # path to your ephemeris
MU = 398600.4418e9  # Earth's mu [m^3/s^2]
Re = 6371e3
G0 = 9.80665

DT = 1.0  # integration time step (s)

# Spacecraft mass parameters
DRY_MASS = 12.0          # kg
PROPELLANT_MASS = 2.0    # kg
M0 = DRY_MASS + PROPELLANT_MASS

# Finite-burn thruster parameters (global throttle scales force linearly)
THRUST_PER_THRUSTER_N = 0.5   # N (each thruster)
ISP = 300.0                   # s
# thruster mounting: list of (r_body_m, dir_body_unit)
THRUSTER_CONFIG = [
    (np.array([0.15, 0.10, 0.0]), np.array([1.0, 0.0, 0.0])),
    (np.array([0.15, -0.10, 0.0]), np.array([1.0, 0.0, 0.0])),
    (np.array([-0.15, 0.10, 0.0]), np.array([-1.0, 0.0, 0.0])),
    (np.array([-0.15, -0.10, 0.0]), np.array([-1.0, 0.0, 0.0])),
    (np.array([0.0, 0.0, 0.08]), np.array([0.0, 0.0, 1.0])),
    (np.array([0.0, 0.0, -0.08]), np.array([0.0, 0.0, -1.0])),
]

# Reaction wheel parameters (three wheels aligned with body X,Y,Z)
RW_INERTIA = 0.0005  # kg*m^2 (wheel rotor inertia)
RW_MAX_TORQUE = 0.05  # N*m max torque the wheel motor can apply (per wheel)
RW_MAX_SPEED = 6000.0 * 2*math.pi/60.0  # rad/s

# Spacecraft body inertia
I_body = np.diag([0.02, 0.025, 0.015])  # kg*m^2
I_body_inv = np.linalg.inv(I_body)

# Attitude controller gains (PD in body frame)
Kp_att = np.array([1.5, 1.5, 1.2])  # Nm per rad
Kd_att = np.array([0.8, 0.8, 0.6])  # Nm per (rad/s)

# Sensor noise for velocity (m/s)
VEL_MEAS_SIGMA = 0.02

# Planned delta-v (m/s) default
PLANNED_DELTA_V = 5.0  # m/s

# Burn selection: first ascending node by default
USE_ASC_NODE = True

# Integration limits
MAX_BURN_TIME = 3600.0  # s
POST_PROP_TIME = 3600.0  # s
# ------------------------

# ------------------------
# Utilities
# ------------------------
def norm(v):
    return np.linalg.norm(v)

def quat_mul_impl(a,b):
    w0,x0,y0,z0 = a
    w1,x1,y1,z1 = b
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_to_dcm(q):
    q0,q1,q2,q3 = q
    R = np.array([
        [1-2*(q2*q2+q3*q3),   2*(q1*q2 - q0*q3),   2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),   1-2*(q1*q1+q3*q3),   2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),   2*(q2*q3 + q0*q1),   1-2*(q1*q1+q2*q2)]
    ])
    return R

def dcm_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr+1.0) * 2
        q0 = 0.25 * S
        q1 = (R[2,1] - R[1,2]) / S
        q2 = (R[0,2] - R[2,0]) / S
        q3 = (R[1,0] - R[0,1]) / S
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            q0 = (R[2,1] - R[1,2]) / S
            q1 = 0.25 * S
            q2 = (R[0,1] + R[1,0]) / S
            q3 = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            q0 = (R[0,2] - R[2,0]) / S
            q1 = (R[0,1] + R[1,0]) / S
            q2 = 0.25 * S
            q3 = (R[1,2] + R[2,1]) / S
        else:
            S = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            q0 = (R[1,0] - R[0,1]) / S
            q1 = (R[0,2] + R[2,0]) / S
            q2 = (R[1,2] + R[2,1]) / S
            q3 = 0.25 * S
    q = np.array([q0,q1,q2,q3])
    return q / (norm(q) + 1e-16)

# ------------------------
# Parse STK/GMAT .e (same format you posted)
# ------------------------
def parse_stk_ephemeris(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    rows = []
    with open(file_path,'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("BEGIN") or s.startswith("END") or s.startswith("ScenarioEpoch") or \
               s.startswith("CentralBody") or s.startswith("CoordinateSystem") or \
               s.startswith("DistanceUnit") or s.startswith("NumberOfEphemerisPoints") or \
               s.startswith("EphemerisTimePosVel") or s.startswith("WrittenBy"):
                continue
            parts = s.split()
            try:
                nums = [float(p) for p in parts[:7]]
                rows.append(nums)
            except:
                continue
    arr = np.array(rows)
    if arr.size == 0:
        return np.zeros((0,)), np.zeros((0,3)), np.zeros((0,3))
    t = arr[:,0]
    r_km = arr[:,1:4]
    v_km_s = arr[:,4:7]
    r_m = r_km * 1000.0
    v_m_s = v_km_s * 1000.0
    return t, r_m, v_m_s

# ------------------------
# Find first ascending node index
# ------------------------
def find_ascending_node(r_hist):
    if r_hist.shape[0] == 0:
        return 0
    z = r_hist[:,2]
    for i in range(len(z)-1):
        if z[i] < 0 and z[i+1] >= 0:
            return i+1
    return 0

# ------------------------
# Force/torque from thrusters
# ------------------------
def thrust_force_and_torque(body_q, thruster_throttles):
    R_bi = quat_to_dcm(body_q)  # body->inertial
    F_inertial = np.zeros(3)
    torque_body = np.zeros(3)
    for (r_b, dir_b), throttle in zip(THRUSTER_CONFIG, thruster_throttles):
        F_body = dir_b / (norm(dir_b)+1e-12) * (THRUST_PER_THRUSTER_N * throttle)  # N in body frame
        F_inert = R_bi @ F_body
        F_inertial += F_inert
        tau_b = np.cross(r_b, F_body)
        torque_body += tau_b
    return F_inertial, torque_body

def attitude_controller_pd(q, w, q_des, w_des):
    q_err = quat_mul_impl(quat_conj(q), q_des)
    ang_err = q_err[1:]
    torque_p = -Kp_att * ang_err
    torque_d = -Kd_att * (w - w_des)
    torque_cmd = torque_p + torque_d
    return torque_cmd

# ------------------------
# Coupled burn simulation
# ------------------------
def run_coupled_burn(r0, v0, q0, w0, ws0, mass0, planned_dv_vec, burn_duration_max=MAX_BURN_TIME, dt=DT):
    n_thr = len(THRUSTER_CONFIG)
    throttles = np.ones(n_thr)
    dv_norm = norm(planned_dv_vec)
    if dv_norm < 1e-9:
        print("[WARN] planned Δv is zero or tiny; using default 5 m/s for visibility.")
        dv_norm = 5.0
        planned_dv_vec = (v0 / (norm(v0)+1e-12)) * dv_norm

    dv_dir = planned_dv_vec / (norm(planned_dv_vec)+1e-12)

    # Desired quaternion: align body X to dv_dir (coarse)
    x_b_des = dv_dir / (norm(dv_dir)+1e-12)
    z_world = np.array([0.0,0.0,1.0])
    y_temp = np.cross(z_world, x_b_des)
    if norm(y_temp) < 1e-6:
        y_temp = np.array([0.0,1.0,0.0])
    y_b_des = y_temp / (norm(y_temp)+1e-12)
    z_b_des = np.cross(x_b_des, y_b_des)
    R_des = np.column_stack((x_b_des, y_b_des, z_b_des))
    q_des = dcm_to_quat(R_des)

    t = 0.0
    max_steps = int(burn_duration_max / dt) + 5
    history = {"t":[], "r":[], "v":[], "q":[], "w":[], "ws":[], "mass":[], "thruster_torque_body":[], "rw_torques":[],"measured_dv":[],"true_dv":[]}

    r = r0.copy()
    v = v0.copy()
    q = q0.copy()
    w = w0.copy()
    ws = ws0.copy()
    mass = mass0

    v_init = v0.copy()
    v_meas_init = v0 + np.random.normal(0, VEL_MEAS_SIGMA, 3)

    for step in range(max_steps):
        F_inertial, tau_thruster_body = thrust_force_and_torque(q, throttles)
        total_thrust = sum([THRUST_PER_THRUSTER_N * th for th in throttles])
        mdot = - total_thrust / (ISP * G0)
        a_thrust_inertial = F_inertial / (mass + 1e-12)
        a_grav = -MU * r / (norm(r)**3)

        torque_cmd_body = attitude_controller_pd(q, w, q_des, np.zeros(3))
        rw_torque_cmd = np.clip(torque_cmd_body, -RW_MAX_TORQUE, RW_MAX_TORQUE)
        body_torque = tau_thruster_body - rw_torque_cmd

        w_dot = I_body_inv @ (body_torque - np.cross(w, I_body @ w))

        Omega = np.array([
            [0.0, -w[0], -w[1], -w[2]],
            [w[0], 0.0, w[2], -w[1]],
            [w[1], -w[2], 0.0, w[0]],
            [w[2], w[1], -w[0], 0.0]
        ])
        q_dot = 0.5 * Omega @ q

        # Integrate attitude & wheels (semi-implicit)
        w = w + w_dot * dt
        q = q + q_dot * dt
        q = q / (norm(q)+1e-12)
        ws_dot = rw_torque_cmd / RW_INERTIA
        ws = ws + ws_dot * dt
        ws = np.clip(ws, -RW_MAX_SPEED, RW_MAX_SPEED)

        # Orbital integrator (simple Euler step)
        a_total = a_grav + a_thrust_inertial
        v = v + a_total * dt
        r = r + v * dt

        mass = mass + mdot * dt
        if mass < DRY_MASS:
            mass = DRY_MASS

        # record
        history["t"].append(t)
        history["r"].append(r.copy())
        history["v"].append(v.copy())
        history["q"].append(q.copy())
        history["w"].append(w.copy())
        history["ws"].append(ws.copy())
        history["mass"].append(mass)
        history["thruster_torque_body"].append(tau_thruster_body.copy())
        history["rw_torques"].append(rw_torque_cmd.copy())

        true_dv = norm(v - v_init)
        v_meas = v + np.random.normal(0, VEL_MEAS_SIGMA, 3)
        measured_dv = norm(v_meas - v_meas_init)
        history["true_dv"].append(true_dv)
        history["measured_dv"].append(measured_dv)

        # stop condition: require a minimum burn time to pass before allowing measured-based stop
        MIN_BURN_TIME_TO_CHECK = max(2.0 * dt, 2.0)  # seconds
        if (t > MIN_BURN_TIME_TO_CHECK) and (measured_dv >= norm(planned_dv_vec) - 1e-6):
            print(f"[INFO] measured Δv target reached at t={t:.1f}s (meas {measured_dv:.4f} m/s, true {true_dv:.4f} m/s)")
            break
        if mass <= DRY_MASS + 1e-9:
            print("[WARN] Propellant exhausted.")
            break

        t += dt

    # finalize history arrays
    for k in history:
        history[k] = np.array(history[k])
    final = {"r": r, "v": v, "q": q, "w": w, "ws": ws, "mass": mass, "t": t}
    return final, history

# ------------------------
# Main
# ------------------------
def main():
    t_hist, r_hist, v_hist = parse_stk_ephemeris(EPHEM_PATH)
    print(f"Loaded ephemeris points: {len(t_hist)}")
    if len(t_hist) == 0:
        print("[ERROR] No ephemeris data found. Check EPHEM_PATH.")
        return

    node_idx = find_ascending_node(r_hist) if USE_ASC_NODE else 0
    print(f"Selected burn index {node_idx}, epoch time {t_hist[node_idx]:.3f} s")

    r_burn = r_hist[node_idx]
    v_burn = v_hist[node_idx]

    # initial attitude: body X aligned roughly with velocity
    v_dir_unit = v_burn / (norm(v_burn)+1e-12)
    z_world = np.array([0.0,0.0,1.0])
    y_tmp = np.cross(z_world, v_dir_unit)
    if norm(y_tmp) < 1e-6:
        y_tmp = np.array([0.0,1.0,0.0])
    y_dir = y_tmp / (norm(y_tmp)+1e-12)
    z_dir = np.cross(v_dir_unit, y_dir)
    R_init = np.column_stack((v_dir_unit, y_dir, z_dir))
    q0 = dcm_to_quat(R_init)
    w0 = np.zeros(3)
    ws0 = np.zeros(3)
    mass0 = M0

    planned_dv_mag = PLANNED_DELTA_V
    planned_dv_vec = (v_burn / (norm(v_burn)+1e-12)) * planned_dv_mag

    print(f"Planned Δv (m/s): {planned_dv_mag:.3f}")

    final, hist = run_coupled_burn(r_burn.copy(), v_burn.copy(), q0.copy(), w0.copy(), ws0.copy(), mass0, planned_dv_vec, burn_duration_max=MAX_BURN_TIME, dt=DT)

    print("\n-- Burn summary --")
    print(f"Time reached: {final['t']:.1f} s")
    print(f"Mass before: {mass0:.3f} kg  after: {final['mass']:.3f} kg  prop used: {mass0 - final['mass']:.4f} kg")
    dv_true = norm(final['v'] - v_burn)
    print(f"True Δv achieved: {dv_true:.4f} m/s")

    # post-burn coast (short)
    steps = int(min(POST_PROP_TIME, 600.0) / DT) + 1  # shorter post-prop for quick visualization
    rs_post = np.zeros((steps,3))
    vs_post = np.zeros((steps,3))
    r = final['r'].copy()
    v = final['v'].copy()
    for k in range(steps):
        rs_post[k] = r.copy()
        vs_post[k] = v.copy()
        a = -MU * r / (norm(r)**3)
        v = v + a * DT
        r = r + v * DT

    # debug prints
    print("History lengths:", {k: (hist[k].shape[0] if k in hist else None) for k in hist})
    if hist["t"].shape[0] == 0:
        print("[WARN] No burn history recorded (hist empty). Check run_coupled_burn logic.")

    # plot orbits
    fig = plt.figure(figsize=(11,9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r_hist[:,0]/1000.0, r_hist[:,1]/1000.0, r_hist[:,2]/1000.0, color='gray', alpha=0.4, label='Ephemeris orig')
    if hist["r"].shape[0] > 0:
        ax.plot(hist["r"][:,0]/1000.0, hist["r"][:,1]/1000.0, hist["r"][:,2]/1000.0, color='orange', label='During burn')
    ax.plot(rs_post[:,0]/1000.0, rs_post[:,1]/1000.0, rs_post[:,2]/1000.0, color='red', label='Post-burn coast')
    ax.scatter([r_burn[0]/1000.0],[r_burn[1]/1000.0],[r_burn[2]/1000.0], color='green', s=80, label='Burn start')
    # Earth
    u,vv = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = (Re/1000.0)*np.cos(u)*np.sin(vv)
    y = (Re/1000.0)*np.sin(u)*np.sin(vv)
    z = (Re/1000.0)*np.cos(vv)
    ax.plot_surface(x,y,z,color='lightblue',alpha=0.5)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Z (km)')
    ax.set_title('Multi-thruster finite burn (reaction-wheel attitude control) - FIXED')
    ax.legend()
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.show()

    # attitude Euler angles
    if hist["q"].shape[0] > 0:
        def quat_to_euler_deg(q):
            w,x,y,z = q
            roll = math.degrees(math.atan2(2*(w*x + y*z), 1-2*(x*x+y*y)))
            pitch = math.degrees(math.asin(max(-1.0,min(1.0,2*(w*y - z*x)))))
            yaw = math.degrees(math.atan2(2*(w*z + x*y), 1-2*(y*y+z*z)))
            return np.array([roll,pitch,yaw])
        euler_hist = np.array([quat_to_euler_deg(q) for q in hist["q"]])
        t_hist_local = hist["t"]
        plt.figure(figsize=(9,4))
        plt.plot(t_hist_local, euler_hist[:,0], label='Roll (deg)')
        plt.plot(t_hist_local, euler_hist[:,1], label='Pitch (deg)')
        plt.plot(t_hist_local, euler_hist[:,2], label='Yaw (deg)')
        plt.xlabel('Time (s)'); plt.ylabel('deg'); plt.title('Attitude during burn'); plt.legend(); plt.grid(True)
        plt.show()
    else:
        print("[INFO] No attitude history to plot.")

    # reaction wheel speeds
    if hist["ws"].shape[0] > 0:
        ws_hist = hist["ws"]
        t_hist_local = hist["t"]
        plt.figure(figsize=(9,4))
        plt.plot(t_hist_local, ws_hist[:,0], label='RW_x (rad/s)')
        plt.plot(t_hist_local, ws_hist[:,1], label='RW_y (rad/s)')
        plt.plot(t_hist_local, ws_hist[:,2], label='RW_z (rad/s)')
        plt.xlabel('Time (s)'); plt.ylabel('rad/s'); plt.title('Reaction wheel speeds'); plt.legend(); plt.grid(True)
        plt.show()
    else:
        print("[INFO] No RW history to plot.")

    # Δv measured vs true
    if hist["t"].shape[0] > 0:
        t_hist_local = hist["t"]
        plt.figure(figsize=(9,4))
        plt.plot(t_hist_local, hist["measured_dv"], label='measured Δv (noisy)')
        plt.plot(t_hist_local, hist["true_dv"], label='true Δv')
        plt.axhline(planned_dv_mag, color='k', linestyle='--', label='planned Δv')
        plt.xlabel('Time (s)'); plt.ylabel('Δv (m/s)'); plt.title('Δv during burn'); plt.legend(); plt.grid(True)
        plt.show()
    else:
        print("[INFO] No Δv history to plot.")

if __name__ == "__main__":
    main()
