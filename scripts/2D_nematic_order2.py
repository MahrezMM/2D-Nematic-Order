#!/usr/bin/env python3
"""
In-plane (2D) nematic order of CNC rods lying flat at the two oil-water
interfaces, using MDAnalysis.

The data file is read only for the bonds: MDAnalysis uses them to group the
beads into molecules ("fragments"), so the rods come from the topology
instead of from counting beads. The coordinates come from the dump.

For every frame:
  - unwrap the rods, so a rod cut by a periodic boundary is put back together
  - rod axis = vector from its first bead to its last bead
  - keep a rod only if it is near an interface (center within ZWIN) AND
    lying flat on it (tilt below FLAT_TILT)
  - build the 2D Q tensor of the kept in-plane angles; S is its largest
    eigenvalue, one value per interface

Needs:  pip install --user MDAnalysis
Usage:  python3 inplane_order_mda.py
"""

import numpy as np
import MDAnalysis as mda

# ------------------------------ settings ------------------------------------
DATA = "cnc.data"            # data file, used for the bonds only
TRAJ = "brush.lammpstrj"     # dump file, used for the coordinates
ATOM_STYLE = "id resid type x y z"   # MUST match the columns of your data file
CNC_TYPES = "1 2 3"          # bead types belonging to a CNC
ZI = 20.0                    # interfaces at z = +ZI and z = -ZI from the box center
ZWIN = 8.0                   # near the interface if the center is this close
FLAT_TILT = 30.0             # flat if the rod tilts less than this out of xy (deg)
SKIP = 0                     # equilibration frames to drop
NBLOCKS = 5                  # blocks used for the error bar
# ----------------------------------------------------------------------------


def order_2d(angles):
    """2D nematic order parameter and director from in-plane angles.

    Q = < 2 u u - I >  with u = (cos t, sin t), a 2x2 symmetric traceless
    matrix. S is its largest eigenvalue, the director is its eigenvector.
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
    n = eigenvectors[:, k]
    director = np.degrees(np.arctan2(n[1], n[0])) % 180.0   # n and -n are equal
    return eigenvalues[k], director


def block_error(x):
    """Error bar from NBLOCKS block averages (frames are correlated)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < NBLOCKS:
        return np.nan
    means = [b.mean() for b in np.array_split(x, NBLOCKS)]
    return np.std(means, ddof=1) / np.sqrt(NBLOCKS)


def main():
    # the universe holds the topology (from DATA) and the trajectory (TRAJ)
    u = mda.Universe(DATA, TRAJ, atom_style=ATOM_STYLE, format="LAMMPSDUMP")
    cnc = u.select_atoms("type " + CNC_TYPES)
    rods = cnc.fragments            # one fragment = one bonded molecule = one rod

    print("beads : %d" % len(cnc))
    print("rods  : %d  (sizes %s)"
          % (len(rods), sorted(set(len(r) for r in rods))))
    print("frames: %d" % len(u.trajectory))

    rows = []
    for ts in u.trajectory[SKIP:]:
        Lz = ts.dimensions[2]
        cnc.unwrap(compound="fragments")     # reassemble rods across boundaries

        angles = {+1: [], -1: []}
        n_near = 0

        for rod in rods:
            pos = rod.positions
            d = pos[-1] - pos[0]             # end-to-end vector
            axis = d / np.linalg.norm(d)

            # z of the rod center, measured from the middle of the box
            # (folding first makes this independent of where the box starts)
            z = (pos[:, 2].mean() % Lz) - 0.5 * Lz

            # filter 1: near one of the two interfaces?
            side = +1 if abs(z - ZI) <= abs(z + ZI) else -1
            if abs(z - side * ZI) > ZWIN:
                continue
            n_near = n_near + 1

            # filter 2: lying flat on it?
            tilt = np.degrees(np.arcsin(min(1.0, abs(axis[2]))))
            if tilt > FLAT_TILT:
                continue

            angles[side].append(np.arctan2(axis[1], axis[0]))

        S = {}
        director = {}
        for side in (+1, -1):
            if len(angles[side]) >= 2:
                S[side], director[side] = order_2d(angles[side])
            else:
                S[side], director[side] = np.nan, np.nan

        rows.append([S[+1], S[-1], director[+1], director[-1],
                     n_near, len(angles[+1]) + len(angles[-1])])

    rows = np.array(rows, float)
    S_mean = np.nanmean(rows[:, 0:2], axis=1)    # average of the two interfaces

    print("\nsettings  : ZI=+-%.1f  ZWIN=%.1f  FLAT_TILT=%.0f  SKIP=%d"
          % (ZI, ZWIN, FLAT_TILT, SKIP))
    print("rods used : %.1f near an interface, %.1f flat (used) per frame"
          % (rows[:, 4].mean(), rows[:, 5].mean()))
    print("S2D (+z)  : %.3f" % np.nanmean(rows[:, 0]))
    print("S2D (-z)  : %.3f" % np.nanmean(rows[:, 1]))
    print("S2D (avg) : %.3f +/- %.3f" % (np.nanmean(S_mean), block_error(S_mean)))

    table = np.column_stack([np.arange(len(rows)), S_mean, rows])
    np.savetxt("inplane_perframe.csv", table, delimiter=",", fmt="%g",
               header="frame,S2D,S2D_plus,S2D_minus,dir_plus,dir_minus,"
                      "n_near,n_flat",
               comments="")
    print("\nwrote inplane_perframe.csv")


if __name__ == "__main__":
    main()
