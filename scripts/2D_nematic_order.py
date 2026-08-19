#!/usr/bin/env python3
"""
In-plane (2D) nematic order of CNC rods lying flat at the two oil-water
interfaces (z = +ZI and z = -ZI).
i
For every frame:
  - keep the CNC beads and cut them into rods of BEADS_PER_ROD beads
  - the rod axis is the vector from its first bead to its last bead
  - a rod is kept only if it is BOTH near an interface (center within ZWIN)
    AND lying flat on it (tilt out of the xy plane below FLAT_TILT)
  - the kept axes are projected on xy and the 2D Q tensor is built
    explicitly; S is its largest eigenvalue, one value per interface

Usage:  python3 inplane_order.py                 # uses TRAJ below
        python3 inplane_order.py other.lammpstrj
"""

import sys
import numpy as np

# ------------------------------ settings ------------------------------------
TRAJ = "brush.lammpstrj"
BEADS_PER_ROD = 140          # Nh = 20 -> 7*20 beads per rod
CNC_TYPES = [1, 2, 3]        # bead types belonging to a CNC
ZI = 20.0                    # interfaces at z = +ZI and z = -ZI
ZWIN = 8.0                   # near the interface if the center is this close
FLAT_TILT = 30.0             # flat if the rod tilts less than this out of xy (deg)
SKIP = 0                     # equilibration frames to drop
NBLOCKS = 5                  # blocks used for the error bar
# ----------------------------------------------------------------------------


def read_frames(fname):
    """Read a LAMMPS dump one frame at a time -> box, lo, types, xyz, wrapped."""
    f = open(fname)
    while True:
        line = f.readline()
        if not line:
            return
        if not line.startswith("ITEM: TIMESTEP"):
            continue
        f.readline()                                     # timestep number
        f.readline()                                     # "NUMBER OF ATOMS"
        n = int(f.readline())
        f.readline()                                     # "BOX BOUNDS"
        bounds = np.array([f.readline().split()[:2] for _ in range(3)], float)
        cols = f.readline().split()[2:]                  # "ATOMS id type x y z"
        data = np.array([f.readline().split() for _ in range(n)], float)
        data = data[np.argsort(data[:, cols.index("id")])]   # sort by atom id
        wrapped = "xu" not in cols                       # xu/yu/zu = unwrapped
        names = ["x", "y", "z"] if wrapped else ["xu", "yu", "zu"]
        lo, hi = bounds[:, 0], bounds[:, 1]
        yield (hi - lo, lo,
               data[:, cols.index("type")].astype(int),
               data[:, [cols.index(c) for c in names]],
               wrapped)


def order_2d(angles):
    """2D nematic order parameter and director from in-plane angles.

    The 2D Q tensor is  Q = < 2 u u - I >  with u = (cos t, sin t).
    It is a 2x2 symmetric traceless matrix. S is its largest eigenvalue,
    and the director is the eigenvector that belongs to it.
        S = 0 -> random in-plane orientations
        S = 1 -> all rods parallel
    """
    Q = np.zeros((2, 2))
    for t in angles:
        u = np.array([np.cos(t), np.sin(t)])
        Q = Q + 2.0 * np.outer(u, u) - np.eye(2)
    Q = Q / len(angles)

    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    k = np.argmax(eigenvalues)
    S = eigenvalues[k]
    n = eigenvectors[:, k]
    director = np.degrees(np.arctan2(n[1], n[0])) % 180.0   # n and -n are the same
    return S, director


def frame_order(box, lo, types, xyz, wrapped):
    """One frame -> S(+z), S(-z), dir(+z), dir(-z), n_rods, n_near, n_flat."""
    xyz = xyz[np.isin(types, CNC_TYPES)]
    nrods = len(xyz) // BEADS_PER_ROD
    angles = {+1: [], -1: []}
    n_near = 0

    for k in range(nrods):
        first = xyz[k * BEADS_PER_ROD]
        last = xyz[(k + 1) * BEADS_PER_ROD - 1]

        # axis = end-to-end vector; if the rod is cut by a periodic boundary
        # the minimum image brings the last bead back next to the first one
        d = last - first
        if wrapped:
            d = d - box * np.round(d / box)
        u = d / np.linalg.norm(d)
        center_z = first[2] + 0.5 * d[2]

        # filter 1: is the rod near one of the two interfaces?
        z = (center_z - lo[2]) % box[2] + lo[2]          # fold z into the box
        side = +1 if abs(z - ZI) <= abs(z + ZI) else -1  # nearer interface
        if abs(z - side * ZI) > ZWIN:
            continue
        n_near = n_near + 1

        # filter 2: is it lying flat on that interface?
        tilt = np.degrees(np.arcsin(min(1.0, abs(u[2]))))
        if tilt > FLAT_TILT:
            continue

        angles[side].append(np.arctan2(u[1], u[0]))

    S = {}
    director = {}
    for side in (+1, -1):
        if len(angles[side]) >= 2:
            S[side], director[side] = order_2d(angles[side])
        else:
            S[side], director[side] = np.nan, np.nan

    n_flat = len(angles[+1]) + len(angles[-1])
    return S[+1], S[-1], director[+1], director[-1], nrods, n_near, n_flat


def block_error(x):
    """Error bar from NBLOCKS block averages (frames are correlated)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < NBLOCKS:
        return np.nan
    means = [b.mean() for b in np.array_split(x, NBLOCKS)]
    return np.std(means, ddof=1) / np.sqrt(NBLOCKS)


def main():
    traj = sys.argv[1] if len(sys.argv) > 1 else TRAJ

    rows = []                                # one row per frame
    for i, frame in enumerate(read_frames(traj)):
        if i >= SKIP:
            rows.append(frame_order(*frame))
    rows = np.array(rows, float)

    S_plus, S_minus = rows[:, 0], rows[:, 1]
    S = np.nanmean(rows[:, 0:2], axis=1)     # average of the two interfaces

    print("file      : %s" % traj)
    print("settings  : ZI=+-%.1f  ZWIN=%.1f  FLAT_TILT=%.0f  SKIP=%d"
          % (ZI, ZWIN, FLAT_TILT, SKIP))
    print("frames    : %d" % len(rows))
    print("rods      : %.1f total, %.1f near an interface, %.1f flat (used)"
          % (rows[:, 4].mean(), rows[:, 5].mean(), rows[:, 6].mean()))
    print("S2D (+z)  : %.3f" % np.nanmean(S_plus))
    print("S2D (-z)  : %.3f" % np.nanmean(S_minus))
    print("S2D (avg) : %.3f +/- %.3f" % (np.nanmean(S), block_error(S)))

    table = np.column_stack([np.arange(len(rows)), S, rows])
    np.savetxt("inplane_perframe.csv", table, delimiter=",", fmt="%g",
               header="frame,S2D,S2D_plus,S2D_minus,dir_plus,dir_minus,"
                      "n_rods,n_near,n_flat",
               comments="")
    print("\nwrote inplane_perframe.csv")


if __name__ == "__main__":
    main()
