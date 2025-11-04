import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
file_path = "../data/NavSat.e"   # path to your GMAT .e file
mu_earth = 398600.4418  # km^3/s^2
R_earth = 6371.0        # km

# ---------------------------------------------------------------------------
# STEP 1: LOAD GMAT EPHEMERIS
# ---------------------------------------------------------------------------
def load_gmat_ephemeris(file_path):
    """
    Loads GMAT ephemeris file with columns:
    Time, X, Y, Z, VX, VY, VZ
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Skip header lines until we reach the numeric data
    data_lines = [l.strip() for l in lines if l and not l.startswith("#") and not l.startswith("Ephemeris")]

    data = []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 7:
            data.append([float(p) for p in parts[:7]])

    df = pd.DataFrame(data, columns=["Time", "X", "Y", "Z", "VX", "VY", "VZ"])
    return df


# ---------------------------------------------------------------------------
# STEP 2: COMPUTE ORBITAL ELEMENTS (for first data point)
# ---------------------------------------------------------------------------
def rv_to_orbital_elements(r_vec, v_vec, mu=mu_earth):
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    e_vec = (np.cross(v_vec, h_vec) / mu) - (r_vec / r)
    e = np.linalg.norm(e_vec)
    a = 1 / ((2 / r) - (v ** 2 / mu))
    i = np.degrees(np.arccos(h_vec[2] / h))

    # Node line
    K = np.array([0, 0, 1])
    n_vec = np.cross(K, h_vec)
    n = np.linalg.norm(n_vec)
    if n != 0:
        Omega = np.degrees(np.arccos(n_vec[0] / n))
        if n_vec[1] < 0:
            Omega = 360 - Omega
    else:
        Omega = 0

    # Argument of perigee
    if n != 0 and e > 1e-6:
        omega = np.degrees(np.arccos(np.dot(n_vec, e_vec) / (n * e)))
        if e_vec[2] < 0:
            omega = 360 - omega
    else:
        omega = 0

    # True anomaly
    if e > 1e-6:
        nu = np.degrees(np.arccos(np.dot(e_vec, r_vec) / (e * r)))
        if np.dot(r_vec, v_vec) < 0:
            nu = 360 - nu
    else:
        nu = 0

    return a, e, i, Omega, omega, nu


# ---------------------------------------------------------------------------
# STEP 3: MAIN SCRIPT
# ---------------------------------------------------------------------------
def main():
    df = load_gmat_ephemeris(file_path)

    print("\n✅ Ephemeris loaded successfully.")
    print(f"Total points: {len(df)}")

    # Convert to numpy arrays
    r = df[["X", "Y", "Z"]].to_numpy()
    v = df[["VX", "VY", "VZ"]].to_numpy()

    # Compute orbital elements for the first point
    a, e, i, Omega, omega, nu = rv_to_orbital_elements(r[0], v[0])

    print("\n📊 Initial Orbital Elements:")
    print(f"Semi-major axis (a): {a:.2f} km")
    print(f"Eccentricity (e): {e:.4f}")
    print(f"Inclination (i): {i:.2f}°")
    print(f"RAAN (Ω): {Omega:.2f}°")
    print(f"Argument of Perigee (ω): {omega:.2f}°")
    print(f"True Anomaly (ν): {nu:.2f}°")

    # -----------------------------------------------------------------------
    # STEP 4: 3D ORBIT PLOT
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r[:, 0], r[:, 1], r[:, 2], color='dodgerblue', label='Orbit Path', linewidth=1.5)

    # Draw Earth
    u, v_ang = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = R_earth * np.cos(u) * np.sin(v_ang)
    y = R_earth * np.sin(u) * np.sin(v_ang)
    z = R_earth * np.cos(v_ang)
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.6, zorder=0)

    # Labels
    ax.set_title("3D Orbit Trajectory (GMAT Ephemeris)", fontsize=12)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.legend()
    ax.grid(True)
    ax.set_box_aspect([1,1,1])

    plt.show()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
