import os, random, hashlib, math, itertools, traceback, time, sqlite3
import networkx as nx
import numpy as np
import multiprocessing as mp
from joblib.externals.loky import cpu_count
from rdkit import Chem, RDLogger, rdBase
RDLogger.DisableLog('rdApp.info')
RDLogger.DisableLog('rdApp.warning')
from rdkit.Chem import AllChem, Draw, rdmolops, rdDepictor, rdchem
from ase import Atoms, io
from ase.visualize import view
from ase.db import connect
from scipy.spatial import KDTree

# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================
DB_CONN=None
# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "mol_0D.db")
# --- Target ---
N_MOLECULES = 203000
# --- Generation Settings ---
MIN_RINGS = 4
MAX_RINGS = 60
# --- Defect Probabilities ---
PROB_SW = 0.10  # Stone-Wales
PROB_SV = 0.15  # Single Vacancy
PROB_DV = 0.20  # Double Vacancy
PROB_EDGE = 0.10  # Edge Defect
MAX_DEFECT_TYPES = 4
# --- Heteroatom Substitution Probabilities ---
PROB_N = 0.10
PROB_O = 0.04
PROB_B = 0.10
PROB_S = 0.04
MAX_SUB_RATIO = 0.35
# --- Parallel Execution ---
TASK_TIMEOUT = 300
GRACE_PERIOD = 2.0
MAX_PARALLEL = max(1, cpu_count() - 2)
# --- Geometry Constants ---
BOND_LENGTH = 1.42
# --- Symmetry-preserved trimming probability ---
CORE_TRIM_P = 0.4

# Growth in multiple stages (fractions of target size)
GROW_STAGE_FRACS = [0.12, 0.24, 0.36, 0.50, 0.65, 0.80, 1.00]
# If grown too much: len(centers) > target * this factor -> trigger symmetric shrink
GROW_OVERFLOW_FACTOR = 1.15
# When shrinking, remove this fraction of current hexes (symmetric way)
GROW_SHRINK_FRACTION = 0.15
# Margin for fill_inside_axial (how many layers beyond boundary get filled)
GROW_FILL_MARGIN = 1
# Max idle iterations when trying to expand frontier
GROW_FRONTIER_IDLE_LIMIT = 2000
# Symmetric wights for C2,C3,C6,mirror,asymmetric
SYM_WIGHTS = [0.120, 0.150, 0.470, 0.240, 0.020]

os.makedirs(SCRIPT_DIR, exist_ok=True)
# =============================================================================
# SECTION 2: HEXAGONAL MATH UTILITIES
# =============================================================================
def axial_to_cartesian(q, r, size=1.42):
    """Converts hexagonal axial coordinates (q, r) to Cartesian coordinates."""
    x = size * math.sqrt(3) * (q + r / 2.0)
    y = size * 3.0 / 2.0 * r
    return (x, y)
def hex_corner(center, size, i):
    """Returns the Cartesian coordinates of the i-th corner of a hexagon."""
    (cx, cy) = center
    angle_deg = 60 * i + 30
    angle_rad = math.radians(angle_deg)
    return (cx + size * math.cos(angle_rad), cy + size * math.sin(angle_rad))
def axial_to_cube_frac(q, r):
    """Allow q,r to be floats; return cube coords (x,y,z) with x+y+z=0."""
    x = q
    z = r
    y = -x - z
    return (x, y, z)
def cube_to_axial_round(x, y, z):
    """
    Standard nearest-hex rounding in cube coords, then project to axial.
    Ensures returned (q,r) are integers on the lattice.
    """
    rx, ry, rz = round(x), round(y), round(z)
    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)

    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return int(rx), int(rz)
def cart_to_axial(x, y, size=1.42):
    """
    Inverse of:
        x = size*sqrt(3)*(q + r/2)
        y = size*(3/2)*r
    then do nearest-cube rounding to keep on lattice
    """
    qf = ((x * math.sqrt(3) / 3) - (y / 3)) / size
    rf = (2 * y / 3) / size
    zf = -qf - rf
    rq = round(qf); rr = round(rf); rz = round(zf)
    dq = abs(rq - qf); dr = abs(rr - rf); dz = abs(rz - zf)
    if dq > dr and dq > dz:
        rq = -rr - rz
    elif dr > dz:
        rr = -rq - rz
    # else rz is largest; rz is implied by rq,rr in axial
    return (rq, rr)
def rot60_cube(x, y, z):
    """+60 degree rotation in cube coords: (x,y,z) -> (-z, -x, -y)"""
    return (-z, -x, -y)
def rot2(v, deg):
    """2D rotation of vector v about origin."""
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return (c*v[0] - s*v[1], s*v[0] + c*v[1])
def neighbors_axial(q, r):
    """Returns 6 neighbors in (q,r) coords."""
    offsets = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    return [(q + dq, r + dr) for dq, dr in offsets]

# =============================================================================
# SECTION 3: SYMMETRY ENGINE
# =============================================================================
def mirror2d_vertical(v):
    return (-v[0], v[1])

TRANSFORM_OPS = {
    'C6': [lambda v,ang=ang: rot2(v, ang) for ang in (0, 60, 120, 180, 240, 300)],
    'C3': [lambda v,ang=ang: rot2(v, ang) for ang in (0, 120, 240)],
    'C2': [lambda v,ang=ang: rot2(v, ang) for ang in (0, 180)],
    'mirror': [lambda v: v, mirror2d_vertical],
    'asymmetric': [lambda v: v],
    'C1': [lambda v: v],
}
MIRROR_CANDIDATE_ANGLES = (0, 60, 120, 180, 240, 300)
def mirror_about_theta(v, theta_deg):
    v1 = rot2(v, -theta_deg)
    v2 = mirror2d_vertical(v1)
    return rot2(v2, theta_deg)
def get_symmetric_equivalents(pos, symmetry, pivot_ax=None):
    """
    pos : (q,r)
        axial coordinates
    symmetry : str
        {'C6','C3','C2','mirror','asymmetric','C1'}
    pivot_ax : (qp, rp) | None
        pivot in axial coords; float allowed
        None => use (0,0) => legacy origin-based symmetry
    """
    q, r = pos
    if pivot_ax is None:
        pivot_ax = (0.0, 0.0)

    # cube of pivot & pos
    px, py, pz = axial_to_cube_frac(pivot_ax[0], pivot_ax[1])
    x,  y,  z  = axial_to_cube_frac(q, r)

    # pos vector relative to pivot
    vx, vy, vz = (x - px, y - py, z - pz)

    # symmetry group → indices
    if symmetry == 'C6':
        ks = (0,1,2,3,4,5)
    elif symmetry == 'C3':
        ks = (0,2,4)
    elif symmetry == 'C2':
        ks = (0,3)
    elif symmetry == 'mirror':   # identity + mirror
        ks = (0,1)
    else:
        # asymmetric / C1
        return [(q, r)]
    out = []
    for idx, k in enumerate(ks):

        if symmetry in ('C6','C3','C2'):
            # rotate k × 60°
            tx, ty, tz = vx, vy, vz
            for _ in range(k):
                tx, ty, tz = rot60_cube(tx, ty, tz)

        elif symmetry == 'mirror':
            if idx == 0:
                tx, ty, tz = vx, vy, vz
            else:
                # axial vertical mirror: (q,r) -> (-q-r, r)
                tx, ty, tz = (-vx - vz, vx, vz)

        else:
            tx, ty, tz = vx, vy, vz

        oq, orr = cube_to_axial_round(px + tx, py + ty, pz + tz)
        out.append((oq, orr))
    return out
def infer_pivot_ax(centers, symmetry):
    """
    Heuristics:
      - Try hex-center lattice node (nearest to centroid)
      - For C2/mirror: also try edge midpoints between neighboring centers
      - For C3:       also try triangle centers of 3-cliques
    Pick the pivot that maximizes self-consistency (# of centers mapped back into set).
    Returns (q_pivot, r_pivot) which may be fractional (like 0.5 or 2/3).
    """
    if not centers:
        return (0.0, 0.0)
    # candidate 1: nearest lattice node to centroid
    cq = sum(q for q,_ in centers)/len(centers)
    cr = sum(r for _,r in centers)/len(centers)
    cand = [(round(cq), round(cr))]
    # neighbors (axial)
    NBR = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    S = set(centers)
    # candidate 2: C2/mirror — edge midpoints
    if symmetry in ('C2','mirror'):
        for (q,r) in list(S):
            for dq,dr in NBR:
                t = (q+dq, r+dr)
                if t in S:
                    cand.append(( (q+t[0])/2.0, (r+t[1])/2.0 ))
    # candidate 3: C3 — triangle centers
    if symmetry == 'C3':
        lst = list(S)
        L = min(len(lst), 200)  # cap for speed
        for i in range(L):
            (q1,r1) = lst[i]
            for dq,dr in NBR:
                q2,r2 = q1+dq, r1+dr
                if (q2,r2) not in S: continue
                # pick a third sharing neighbor with q2
                for dq2,dr2 in NBR:
                    q3,r3 = q2+dq2, r2+dr2
                    if (q3,r3) in S and (q3,r3)!=(q1,r1):
                        cand.append(((q1+q2+q3)/3.0, (r1+r2+r3)/3.0))
    # score candidates
    def score(piv):
        hit = 0
        for c in S:
            for eq in get_symmetric_equivalents(c, symmetry, piv):
                if eq in S:
                    hit += 1
        return hit

    best = max(cand, key=score)
    return best
def _get_orbital_atoms(G, ref_node, symmetry, axial_map, pivot_ax=None):
    """
    Return the symmetry orbit (list of node ids) of `ref_node`.
    """
    if ref_node not in G:
        return []
    # -------- pivot --------
    if pivot_ax is None:
        px, py = 0.0, 0.0
    else:
        px, py = axial_to_cartesian(pivot_ax[0], pivot_ax[1])

    # -------- build / reuse XY->node index --------
    # cache key depends only on number_of_nodes; invalidated by _invalidate_hex_cache
    if not hasattr(G, "_xy_index_cache") or \
       G._xy_index_cache.get("stamp") != G.number_of_nodes():
        idx = {}
        for n in G.nodes():
            x, y = G.nodes[n]['xy']
            idx[(round(x, 6), round(y, 6))] = n
        G._xy_index_cache = {
            "map": idx,
            "stamp": G.number_of_nodes()
        }
    idx_map = G._xy_index_cache["map"]
    # -------- symmetry ops --------
    if symmetry == 'C6':
        ops = [0, 60, 120, 180, 240, 300]
    elif symmetry == 'C3':
        ops = [0, 120, 240]
    elif symmetry == 'C2':
        ops = [0, 180]
    elif symmetry == 'mirror':
        # --- choose best_theta from lattice centers ---
        centers = set(axial_map.values())
        # pre-compute all center Cartesian positions
        center_cart_set = set()
        for (cq, cr) in centers:
            cx0, cy0 = axial_to_cartesian(cq, cr)
            center_cart_set.add((round(cx0, 3), round(cy0, 3)))
        def score(theta):
            """Return how many centers map onto centers under mirror w.r.t. θ."""
            hit = 0
            for (cq, cr) in centers:
                cx, cy = axial_to_cartesian(cq, cr)
                vx, vy = (cx - px, cy - py)

                v1x, v1y = rot2((vx, vy), -theta)   # rotate to canonical
                v2x, v2y = (-v1x, v1y)              # reflect in vertical
                nx, ny   = rot2((v2x, v2y), theta)  # rotate back

                wx = px + nx
                wy = py + ny
                key = (round(wx, 3), round(wy, 3))
                if key in center_cart_set:
                    hit += 1
            return hit
        # pick θ that gives best self-consistency
        best_theta = max(
            ((score(t), t) for t in (0, 60, 120, 180, 240, 300)),
            key=lambda x: x[0]
        )[1]
        ops = ['mirror', 0]   # first mirror, then identity
    else:
        ops = [0]             # asymmetric
    # -------- transform ref node --------
    cx, cy = G.nodes[ref_node]['xy']
    rel0 = (cx - px, cy - py)
    orbit = []

    for op in ops:
        if op == 'mirror':
            # mirror = R^-θ, reflect, R^θ
            v1 = rot2(rel0, -best_theta)
            v2 = (-v1[0],  v1[1])      # vertical reflection
            rel = rot2(v2, best_theta)
        else:
            rel = rot2(rel0, op)

        tx = px + rel[0]
        ty = py + rel[1]

        # ---- map back (x,y) → node ----
        # first try 1e-6 rounding
        key = (round(tx, 6), round(ty, 6))
        tgt = idx_map.get(key)
        if tgt is None:
            # fallback with 1e-5 rounding
            key2 = (round(tx, 5), round(ty, 5))
            tgt = idx_map.get(key2)

        if tgt is None:
            # mapping failed -> orbit invalid
            return []

        orbit.append(tgt)
    return list(dict.fromkeys(orbit))
# =============================================================================
# SECTION 4: FLAKE GENERATION & GRAPH BUILDING
# =============================================================================
def _ensure_rng(rng):
    return rng if rng is not None else random.Random()
def grow_random_symmetric(steps, symmetry, start_mode='auto', rng=None):
    """
    Symmetric growth with explicit start modes and correct pivot handling.
    start_mode : str
        One of:
          - 'auto'        : (default) choose seed by symmetry
              · C2/mirror -> random choice of 'hex_center' or 'naphthalene' or 'biphenyl'
              · C3        -> random choice of 'hex_center' or 'phenalene'
              · others    -> 'hex_center'
    """
    rng = _ensure_rng(rng)
    # axial neighbor offsets (pointy-top lattice)
    NBR = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]

    # ---------- seed libraries ----------
    def seed_hex_center():
        return {(0,0)}

    def seed_naphthalene():
        return {(0,0),(0,1)}

    def seed_phenalene():
        return {(0,0),(0,1),(1,0)}

    # ---------- decide seed by start_mode / symmetry ----------
    mode = start_mode
    if mode == 'auto':
        if symmetry in ('C2', 'mirror'):
            mode = rng.choice(['hex_center', 'naphthalene'])
        elif symmetry == 'C3':
            mode = rng.choice(['hex_center', 'phenalene'])
        else:
            mode = 'hex_center'

    if mode == 'hex_center':
        centers = set(seed_hex_center())
    elif mode == 'naphthalene':
        centers = set(seed_naphthalene())
    elif mode == 'phenalene':
        centers = set(seed_phenalene())
    else:
        # fallback safe default
        centers = set(seed_hex_center())

    # ---------- compute pivot in axial (centroid of seed block) ----------
    if symmetry == 'C3' and mode == 'phenalene':
        size = 1.42

        def sixverts(ax):
            cx, cy = axial_to_cartesian(ax[0], ax[1], size)
            return [hex_corner((cx, cy), size, i) for i in range(6)]

        centers_list = list(centers)
        V = [sixverts(ax) for ax in centers_list]

        def quantize(p, dec=4):
            return (round(p[0], dec), round(p[1], dec))

        sets = [set(quantize(p) for p in vv) for vv in V]
        common_xy = list(sets[0] & sets[1] & sets[2])

        def cart_to_axial_float_for_pivot(x, y, size=1.42):
            qf = ((x * math.sqrt(3) / 3) - (y / 3)) / size
            rf = (2 * y / 3) / size
            return (qf, rf)

        if common_xy:
            vx, vy = common_xy[0]
            pivot_ax = cart_to_axial_float_for_pivot(vx, vy, size=size)
        else:
            NBR = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
            S = set(centers)
            found = None
            for (q1, r1) in list(S):
                for dq, dr in NBR:
                    q2, r2 = q1 + dq, r1 + dr
                    if (q2, r2) not in S:
                        continue
                    for dq2, dr2 in NBR:
                        q3, r3 = q2 + dq2, r2 + dr2
                        if (q3, r3) in S and (q3, r3) != (q1, r1):
                            Vs = [set(quantize(p) for p in sixverts(ax))
                                  for ax in [(q1, r1), (q2, r2), (q3, r3)]]
                            inter = list(Vs[0] & Vs[1] & Vs[2])
                            if inter:
                                found = inter[0]
                                break
                    if found: break
                if found: break
            if found:
                vx, vy = found
                pivot_ax = cart_to_axial_float_for_pivot(vx, vy, size=size)
            else:
                pivot_ax = infer_pivot_ax(centers, symmetry)

    elif symmetry in ('C2', 'mirror', 'C3'):
        pivot_ax = infer_pivot_ax(centers, symmetry)

    else:
        # asymmetric / C1
        pq = [c[0] for c in centers]
        pr = [c[1] for c in centers]
        pivot_ax = (sum(pq) / len(pq), sum(pr) / len(pr))

    # pivot Cartesian
    px, py = axial_to_cartesian(pivot_ax[0], pivot_ax[1])

    # ---------- orbit relative to pivot ----------
    ops = TRANSFORM_OPS.get(symmetry, [lambda v: v])

    if symmetry == 'mirror':
        def score_theta(theta):
            hit = 0
            for (q, r) in centers:
                cx, cy = axial_to_cartesian(q, r)
                vx, vy = (cx - px, cy - py)
                # identity
                nx1, ny1 = px + vx, py + vy
                aq1, ar1 = cart_to_axial(nx1, ny1)
                if (aq1, ar1) in centers:
                    hit += 1
                # mirrored
                mx, my = mirror_about_theta((vx, vy), theta)
                nx2, ny2 = px + mx, py + my
                aq2, ar2 = cart_to_axial(nx2, ny2)
                if (aq2, ar2) in centers:
                    hit += 1
            return hit

        best_theta = max(
            ((score_theta(t), t) for t in MIRROR_CANDIDATE_ANGLES),
            key=lambda x: x[0]
        )[1]

        def mirror_op(v):
            return mirror_about_theta(v, best_theta)

        ops = [lambda v: v, mirror_op]

    def orbit(ax_pos):
        """
        Return symmetric images of axial ax_pos under 'symmetry' about the pivot.
        Uses discrete cube rounding to keep on lattice. Enforce full, distinct orbit.
        """
        exp_len_map = {'C6': 6, 'C3': 3, 'C2': 2, 'mirror': 2}

        if symmetry == 'mirror':
            cx, cy = axial_to_cartesian(ax_pos[0], ax_pos[1])
            vx, vy = (cx - px, cy - py)

            out = set()
            out.add(ax_pos)  # identity

            v1x, v1y = rot2((vx, vy), -best_theta)
            v2x, v2y = (-v1x, v1y)
            mx, my = rot2((v2x, v2y), best_theta)
            nx, ny = px + mx, py + my
            oq, orr = cart_to_axial(nx, ny)
            out.add((oq, orr))
            return out

        px_c, py_c, pz_c = axial_to_cube_frac(pivot_ax[0], pivot_ax[1])
        x, y, z = axial_to_cube_frac(ax_pos[0], ax_pos[1])
        vx, vy, vz = (x - px_c, y - py_c, z - pz_c)

        def rot60_cube_once(x, y, z):
            return (-z, -x, -y)

        if symmetry == 'C6':
            ks = (0, 1, 2, 3, 4, 5)
        elif symmetry == 'C3':
            ks = (0, 2, 4)  # 0°,120°,240°
        elif symmetry == 'C2':
            ks = (0, 3)  # 0°,180°
        else:
            ks = (0,)

        out = []
        for k in ks:
            tx, ty, tz = vx, vy, vz
            for _ in range(k):
                tx, ty, tz = rot60_cube_once(tx, ty, tz)
            oq, orr = cube_to_axial_round(px_c + tx, py_c + ty, pz_c + tz)
            out.append((oq, orr))

        if len(set(out)) != len(ks):
            return set()
        return set(out)

    # ---------- utilities ----------
    def is_conn(C):
        if not C:
            return True
        seen = set()
        stack = [next(iter(C))]
        while stack:
            q, r = stack.pop()
            if (q, r) in seen:
                continue
            seen.add((q, r))
            for dq, dr in NBR:
                t = (q+dq, r+dr)
                if t in C and t not in seen:
                    stack.append(t)
        return len(seen) == len(C)

    def frontier(C):
        F = set()
        for q, r in C:
            for dq, dr in NBR:
                t = (q+dq, r+dr)
                if t not in C:
                    F.add(t)
        return F

    # ---------- growth ----------
    def grow_to(target):
        F = frontier(centers)
        idle = 0
        while len(centers) < target and F and idle < GROW_FRONTIER_IDLE_LIMIT:
            moved = False
            cand = list(F)
            rng.shuffle(cand)
            for p in cand:
                orb = orbit(p)
                # avoid conflicts
                if any(o in centers for o in orb):
                    continue
                # at least one member of the orbit should be adjacent to current set
                connected = any(
                    (o[0]+dq, o[1]+dr) in centers
                    for o in orb for dq,dr in NBR
                )
                if not connected:
                    continue

                centers.update(orb)
                F = frontier(centers)
                moved = True
                if len(centers) >= target:
                    break

            idle = 0 if moved else idle + 1

    # staged growth
    T = int(steps)
    for frac in GROW_STAGE_FRACS:
        grow_to(int(T * frac))

    if len(centers) > T * GROW_OVERFLOW_FACTOR:
        excess = int(len(centers) * GROW_SHRINK_FRACTION)
        F = frontier(centers)
        interior = [p for p in centers if p not in F]
        rng.shuffle(interior)
        removed = 0
        for p in interior:
            orb = orbit(p)
            if not orb.issubset(centers):
                continue
            trial = centers - orb
            if trial and is_conn(trial):
                centers = trial
                removed += len(orb)
            if removed >= excess:
                break

    centers = set(fill_inside_axial(centers, symmetry=symmetry, margin=GROW_FILL_MARGIN, pivot_ax=pivot_ax))
    for (q, r) in centers:
        assert isinstance(q, int) and isinstance(r, int), f"Non-integer axial: {(q, r)}"
    return sorted(centers), pivot_ax
def fill_inside_axial(centers, symmetry='asymmetric', margin=2, pivot_ax=None):
    """
    Fill the interior of centers on axial hex grid so that the flake becomes simply-connected.
    All interior atoms will be degree=3.
    """
    S = set(centers)
    if not S:
        return centers

    qs = [q for q,_ in S]
    rs = [r for _,r in S]
    min_q, max_q = min(qs)-margin, max(qs)+margin
    min_r, max_r = min(rs)-margin, max(rs)+margin

    box = {(q, r)
           for q in range(min_q, max_q + 1)
           for r in range(min_r, max_r + 1)}

    empty = box - S
    from collections import deque
    # Determine flood start = all box boundary empty cells
    rim = set()
    for q in range(min_q, max_q + 1):
        rim.add((q, min_r))
        rim.add((q, max_r))
    for r in range(min_r, max_r + 1):
        rim.add((min_q, r))
        rim.add((max_q, r))

    start = [p for p in rim if p in empty]
    visited_out = set()
    dq = deque(start)

    while dq:
        u = dq.popleft()
        if u in visited_out:
            continue
        visited_out.add(u)

        q, r = u
        for v in neighbors_axial(q, r):
            if v in empty and v not in visited_out:
                dq.append(v)

    # interior = empty but NOT reachable from outside
    interior = empty - visited_out

    # add symmetric orbit of interior
    to_add = set()
    for h in interior:
        if symmetry in ('asymmetric', 'C1'):
            to_add.add(h)
        else:
            for eq in get_symmetric_equivalents(h, symmetry, pivot_ax):
                if (min_q <= eq[0] <= max_q) and (min_r <= eq[1] <= max_r):
                    to_add.add(eq)

    S |= to_add
    return sorted(S)
def symmetric_random_ring_delete(centers, symmetry, pivot_ax, rng):
    """
    Symmetry-preserving ring deletion.
    """
    if not centers:
        return centers
    # --- pivot ---
    if pivot_ax is None:
        cq = sum(q for q, _ in centers) / len(centers)
        cr = sum(r for _, r in centers) / len(centers)
        pivot_ax = (cq, cr)
    px, py, pz = axial_to_cube_frac(pivot_ax[0], pivot_ax[1])
    # --- hex distance from pivot ---
    dist_map = {}
    max_d = 0
    for (q, r) in centers:
        x, y, z = axial_to_cube_frac(q, r)
        d = int(round(max(abs(x - px), abs(y - py), abs(z - pz))))
        dist_map[(q, r)] = d
        max_d = max(max_d, d)
    if max_d <= 1:
        return centers

    # Choose how many inner layers to delete
    n_layers_to_remove = rng.randint(1, max_d - 1)

    centers_set = set(centers)
    unassigned = set(centers)
    orbits = []

    # --- build symmetry orbits ---
    while unassigned:
        c = unassigned.pop()
        equivs = get_symmetric_equivalents(c, symmetry, pivot_ax)
        orbit = [p for p in equivs if p in centers_set]
        for p in orbit:
            unassigned.discard(p)
        orbits.append(orbit)

    # --- inner/outer filtering ---
    new_centers = []
    for orbit in orbits:
        if not orbit:
            continue

        d_orbit = min(dist_map[p] for p in orbit)

        if d_orbit >= n_layers_to_remove:
            # outer shells: always keep
            new_centers.extend(orbit)
        else:
            # inner shells: probabilistic keep
            if rng.random() >= 0.7:
                new_centers.extend(orbit)

    if not new_centers:
        return centers

    new_set = set(new_centers)

    # --- find connected components on hex grid ---
    def get_components(ax_set):
        remaining = set(ax_set)
        comps = []

        while remaining:
            start = remaining.pop()
            comp = {start}
            stack = [start]

            while stack:
                q, r = stack.pop()
                for nq, nr in neighbors_axial(q, r):
                    if (nq, nr) in remaining:
                        remaining.remove((nq, nr))
                        comp.add((nq, nr))
                        stack.append((nq, nr))
            comps.append(comp)

        return comps

    comps = get_components(new_set)

    if len(comps) == 1:
        return sorted(new_set)

    # --- keep only outermost components ---
    comp_dist = []
    for comp in comps:
        m = max(dist_map[p] for p in comp)
        comp_dist.append((m, comp))

    max_outer_dist = max(m for m, _ in comp_dist)

    keep_set = set()
    for m, comp in comp_dist:
        if m == max_outer_dist:
            keep_set |= comp

    if not keep_set:
        keep_set = new_set

    return sorted(keep_set)
def cleanup_fragments(mol):
    fragments = rdmolops.GetMolFrags(mol, asMols=True)
    if len(fragments) <= 1:
        return mol
    largest_frag = None
    max_atoms = 0
    for frag in fragments:
        num_atoms = frag.GetNumAtoms()
        if num_atoms > max_atoms:
            max_atoms = num_atoms
            largest_frag = frag
    return largest_frag
def add_internal_bonds_nx(G, symmetry, axial_map, pivot_ax, max_dist=2.0):
    """
    NetworkX version of add_internal_bonds. Closes internal rings by
    adding new bonds between nodes with degree 2, in a SYMMETRIC way.
    """
    H = G.copy()
    pos = {n: H.nodes[n]['xy'] for n in H.nodes()}

    while True:
        bonds_added_in_this_iteration = False
        degree2_atoms_indices = [n for n, deg in H.degree() if deg == 2]
        if len(degree2_atoms_indices) < 2:
            break

        potential_bonds = []
        for i, j in itertools.combinations(degree2_atoms_indices, 2):
            if not H.has_edge(i, j) and H.degree(i) == 2 and H.degree(j) == 2:
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                if dist < max_dist:
                    potential_bonds.append({'atoms': (i, j), 'dist': dist})

        if not potential_bonds:
            break

        potential_bonds.sort(key=lambda b: b['dist'])

        # Find the highest-priority *seed* bond whose *entire orbit* is valid
        for bond_info in potential_bonds:
            seed_i, seed_j = bond_info['atoms']

            # 1. Check if this seed is still valid (nodes might have been used by a previous,
            #    higher-priority orbit in this loop, although we break, this is a good safety check)
            if H.degree(seed_i) != 2 or H.degree(seed_j) != 2:
                continue

            # 2. Find the symmetric orbit of the bond
            try:
                orb_i = _get_orbital_atoms(H, seed_i, symmetry, axial_map, pivot_ax)
                orb_j = _get_orbital_atoms(H, seed_j, symmetry, axial_map, pivot_ax)
            except Exception:
                continue  # Failed to get orbits

            if len(orb_i) != len(orb_j):
                continue  # Asymmetric orbit, skip

            # 3. Validate the entire orbit
            bonds_to_add = list(zip(orb_i, orb_j))
            all_valid = True
            nodes_in_this_orbit = set()
            for u, v in bonds_to_add:
                # Check nodes exist, are not bonded, degrees are correct (== 2)
                if (not H.has_node(u) or not H.has_node(v) or
                        H.degree(u) != 2 or H.degree(v) != 2 or
                        H.has_edge(u, v) or u == v):
                    all_valid = False
                    break

                # Check for orbital self-intersection (e.g., C2 orbit (i,j) and (j,i))
                if u in nodes_in_this_orbit or v in nodes_in_this_orbit:
                    all_valid = False
                    break
                nodes_in_this_orbit.add(u)
                nodes_in_this_orbit.add(v)

            if not all_valid:
                continue  # This orbit is invalid, try the next-best seed bond

            # 4. Success. Apply the *full* orbit and break to restart the while loop.
            for u, v in bonds_to_add:
                H.add_edge(u, v)

            bonds_added_in_this_iteration = True
            break

        if not bonds_added_in_this_iteration:
            break  # No more bonds (or valid symmetric orbits) can be added.
    return H
def detect_highest_symmetry(G, axial_map, centers=None, pivot_ax=None):
    """Detect actual highest symmetry, allowing a non-origin pivot."""
    nodes = list(G.nodes())
    if not nodes:
        return 'asymmetric'

    # auto infer centers if not provided
    if centers is None:
        centers = list(axial_map.values())

    # --- small helpers (reuse your lattice mapping) ---
    def cart_from_ax(ax):
        return axial_to_cartesian(ax[0], ax[1])

    # vector ops
    vec_ops_map = TRANSFORM_OPS

    # --- choose / infer pivot in axial (float allowed) ---
    if pivot_ax is None:
        # centers as defult pivot
        pq = sum(q for q, _ in centers) / max(1, len(centers))
        pr = sum(r for _, r in centers) / max(1, len(centers))
        pivot_ax = (pq, pr)

    px, py = cart_from_ax(pivot_ax)

    # {(axial, elem)}
    node_sig = set((axial_map[n], G.nodes[n]['elem']) for n in nodes if n in axial_map)

    # mirror rotate：R(θ)^{-1}·MirrorV·R(θ)
    def make_mirror_ops(theta_deg):
        def _mirror(v):
            return mirror_about_theta(v, theta_deg)
        return [lambda v: v, _mirror]
    # check symmetry
    def check_symmetry(sym):
        ops = vec_ops_map.get(sym, [lambda v: v])
        best_theta = 0
        if sym == 'mirror':
            scores = []
            for theta in (0, 60, 120, 180, 240, 300):
                ops_try = make_mirror_ops(theta)
                hit = 0
                for (q, r), elem in node_sig:
                    cx, cy = cart_from_ax((q, r))
                    vx, vy = cx - px, cy - py
                    for op in ops_try:
                        nx, ny = px + op((vx, vy))[0], py + op((vx, vy))[1]
                        aq, ar = cart_to_axial(nx, ny)
                        if ((aq, ar), elem) in node_sig:
                            hit += 1
                scores.append((hit, theta))
            best_theta = max(scores)[1]
            ops = make_mirror_ops(best_theta)

        for (q, r), elem in node_sig:
            cx, cy = cart_from_ax((q, r))
            vx, vy = cx - px, cy - py
            for op in ops:
                nx, ny = px + op((vx, vy))[0], py + op((vx, vy))[1]
                aq, ar = cart_to_axial(nx, ny)
                if ((aq, ar), elem) not in node_sig:
                    return False
        return True

    if check_symmetry('C6'): return 'C6'
    if check_symmetry('C3'): return 'C3'
    if check_symmetry('C2'): return 'C2'
    if check_symmetry('mirror'): return 'mirror'
    return 'asymmetric'
def centers_to_graph(centers, size=1.42):
    """Returns (G, axial_map) """
    G = nx.Graph()
    vertex_map = {}
    node_id_counter = 0
    axial_map = {}  # Node ID -> axial coordinate (q,r).

    def round_coord(pt, decimals=6):
        return (round(pt[0], decimals), round(pt[1], decimals))

    for center_ax in centers:
        center_cart = axial_to_cartesian(center_ax[0], center_ax[1], size)
        verts_cart = [hex_corner(center_cart, size, i) for i in range(6)]

        vert_indices = []
        for v_cart in verts_cart:
            rv_cart = round_coord(v_cart)
            if rv_cart not in vertex_map:
                idx = node_id_counter
                vertex_map[rv_cart] = idx
                G.add_node(idx, xy=v_cart, elem='C', ref_centers=[center_ax])
                axial_map[idx] = center_ax
                node_id_counter += 1
            else:
                idx = vertex_map[rv_cart]
                lst = G.nodes[idx].get('ref_centers', [])
                if center_ax not in lst:
                    lst.append(center_ax)
                G.nodes[idx]['ref_centers'] = lst

            vert_indices.append(vertex_map[rv_cart])

        for i in range(6):
            u, v = vert_indices[i], vert_indices[(i + 1) % 6]
            if not G.has_edge(u, v):
                G.add_edge(u, v)

    # Remove dangling bonds/nodes.
    dangling = [n for n, d in G.degree() if d <= 1]
    while dangling:
        G.remove_nodes_from(dangling)
        for n in dangling:
            axial_map.pop(n, None)
        dangling = [n for n, d in G.degree() if d <= 1]

    return G, axial_map
def _remove_dangling(G):
    """Helper function: Iteratively removes all nodes with degree < 2 until the graph is stable."""
    H = G.copy()
    # Nodes with degree 0 or 1 are considered dangling
    dangling = [n for n, d in H.degree() if d < 2]
    while dangling:
        H.remove_nodes_from(dangling)
        # Removing nodes might create new dangling nodes, so we loop
        dangling = [n for n, d in H.degree() if d < 2]
    return H
# =============================================================================
# SECTION 5: DEFECT APPLICATOR
# =============================================================================
def _invalidate_hex_cache(G):
    if hasattr(G, "_hex_cache"):
        G._hex_cache.clear()

    if hasattr(G, "_xy_index_cache"):
        try:
            delattr(G, "_xy_index_cache")
        except:
            pass
def _hexes(G):
    """Find all 6-membered rings in the graph."""
    try:
        return [c for c in nx.cycle_basis(G) if len(c) == 6]
    except nx.exception.NetworkXNoCycle:
        return []
def apply_symmetric_sw(G, symmetry, axial_map, pivot_ax, rng=None):
    all_hexes = _hexes(G)
    rng = _ensure_rng(rng)
    cand = [(u, v) for u, v in G.edges() if G.degree(u) == 3 and G.degree(v) == 3]
    if not cand:
        return G
    rng.shuffle(cand)

    for c1, c2 in cand:
        c3_set = {w for w in G.neighbors(c1) if w != c2 and
                  any(c1 in c and c2 in c and w in c for c in all_hexes)}
        if not c3_set:
            continue
        c3 = rng.choice(list(c3_set))

        # c4: Adjacent to c2, but not in the ring containing c1-c2-c3.
        avoid = [c for c in _hexes(G) if c1 in c and c2 in c]
        c4_set = {w for w in G.neighbors(c2) if w != c1 and
                  not any(w in c for c in avoid)}
        if not c4_set:
            continue
        c4 = rng.choice(list(c4_set))
        break
    else:
        return G

    # 2. Align orbits.
    c1_orb = _get_orbital_atoms(G, c1, symmetry, axial_map, pivot_ax)
    c2_orb = _get_orbital_atoms(G, c2, symmetry, axial_map, pivot_ax)
    c3_orb = _get_orbital_atoms(G, c3, symmetry, axial_map, pivot_ax)
    c4_orb = _get_orbital_atoms(G, c4, symmetry, axial_map, pivot_ax)
    if not (len(c1_orb) == len(c2_orb) == len(c3_orb) == len(c4_orb)):
        return G

    # 3. Batch fixed rotation.
    H = G.copy()
    for x1, x2, x3, x4 in zip(c1_orb, c2_orb, c3_orb, c4_orb):
        if not (H.has_edge(x1, x2) and H.degree(x1) == H.degree(x2) == 3 and
                H.has_edge(x1, x3) and H.has_edge(x2, x4)):
            continue
        H.remove_edge(x1, x3)
        H.remove_edge(x2, x4)
        H.add_edge(x1, x4)
        H.add_edge(x2, x3)

    _invalidate_hex_cache(H)

    return H
def apply_symmetric_sv(G, symmetry, axial_map, pivot_ax, num_sites=1, rng=None):
    """
    Applies Single Vacancy (SV) defects after a strict selection process.
    1. Finds 'C' atoms with degree 3, whose 3 neighbors are also 'C' and degree 3.
    2. Verifies a symmetric reconstruction pairing exists *before* applying.
    """
    rng = _ensure_rng(rng)
    H = G.copy()

    # 1. Strict Selection: Find all valid "bulk" atoms.
    # A valid seed is 'C', degree 3, and all 3 neighbors are 'C' and degree 3.
    valid_seed_candidates = []
    for node in H.nodes():
        if (H.nodes[node]['elem'] == 'C' and
                H.degree(node) == 3 and
                node in axial_map):

            neighbors = list(H.neighbors(node))
            if len(neighbors) == 3:
                n1, n2, n3 = neighbors
                if (H.has_node(n1) and H.has_node(n2) and H.has_node(n3) and
                        H.nodes[n1]['elem'] == 'C' and H.degree(n1) == 3 and
                        H.nodes[n2]['elem'] == 'C' and H.degree(n2) == 3 and
                        H.nodes[n3]['elem'] == 'C' and H.degree(n3) == 3):
                    valid_seed_candidates.append(node)

    if not valid_seed_candidates:
        print("No valid bulk atoms (deg=3 C, with deg=3 C neighbors) found for SV.")
        return G

    rng.shuffle(valid_seed_candidates)

    # 2. Apply defect orbit by orbit
    removed_nodes = set()
    used = 0

    for seed_atom in valid_seed_candidates:
        if used >= num_sites:
            break
        if seed_atom in removed_nodes:
            continue

        # 3. Verify symmetric reconstruction is possible *before* applying
        seed_nbrs = list(H.neighbors(seed_atom))
        if len(seed_nbrs) != 3: continue  # Should be guaranteed by check above, but for safety

        valid_pairing_found = False
        bonds_to_add = []
        orb_set = frozenset()

        neighbor_pairs = [(seed_nbrs[0], seed_nbrs[1]),
                          (seed_nbrs[1], seed_nbrs[2]),
                          (seed_nbrs[0], seed_nbrs[2])]
        rng.shuffle(neighbor_pairs)

        for seed_a, seed_b in neighbor_pairs:
            try:
                orb_atom_list = _get_orbital_atoms(H, seed_atom, symmetry, axial_map, pivot_ax)
                orb_a = _get_orbital_atoms(H, seed_a, symmetry, axial_map, pivot_ax)
                orb_b = _get_orbital_atoms(H, seed_b, symmetry, axial_map, pivot_ax)
            except Exception:
                continue

            if len(orb_atom_list) == len(orb_a) == len(orb_b):
                valid_pairing_found = True
                bonds_to_add = list(zip(orb_a, orb_b))
                orb_set = frozenset(orb_atom_list)
                break

        if not valid_pairing_found:
            # print(f"Skipping SV at {seed_atom}: No symmetric pairing found.")
            continue

        # 4. Check for overlaps
        nodes_to_remove_now = orb_set
        atoms_in_new_bonds = {n for pair in bonds_to_add for n in pair}

        if (nodes_to_remove_now.isdisjoint(removed_nodes) and
                atoms_in_new_bonds.isdisjoint(nodes_to_remove_now)):

            # 5. Apply the defect
            H.remove_nodes_from(nodes_to_remove_now)
            for u, v in bonds_to_add:
                if H.has_node(u) and H.has_node(v) and not H.has_edge(u, v):
                    H.add_edge(u, v)

            removed_nodes.update(nodes_to_remove_now)
            used += 1

    if used == 0:
        return G

    _invalidate_hex_cache(H)
    return H
#  DOUBLE-VACANCY (DV) DEFECT — SYMMETRIC PIPELINE
def _neighbors_cache(G):
    return {n: list(G.neighbors(n)) for n in G.nodes()}
def _safe_has_nodes(G, nodes):
    return all(G.has_node(x) for x in nodes)
def _edge_exists(G, u, v):
    return G.has_edge(u, v)
def _temp_remove_edge_has_path_len(G, u, v, need_len, cutoff=None):
    if not _edge_exists(G, u, v):
        return False
    G.remove_edge(u, v)
    ok = False
    try:
        try:
            sp = nx.shortest_path_length(G, source=u, target=v)
            ok = (sp == need_len)
        except nx.NetworkXNoPath:
            ok = False
    finally:
        G.add_edge(u, v)
    return ok
def _ring_signature_deg3(G, x, cutoff=10):
    """Return a sorted tuple of 3 ring sizes inferred at degree-3 node x.
    For neighbors y,z of x, ring size ~ shortest_path_length_G_minus_x(y,z) + 2.
    If x is not degree-3 or path missing, return None.
    """
    if x not in G or G.degree(x) != 3:
        return None
    nbrs = list(G.neighbors(x))
    if len(nbrs) != 3:
        return None
    # remove x temporarily
    attrs = dict(G.nodes[x])
    G.remove_node(x)
    sizes = []
    try:
        for i in range(3):
            for j in range(i+1, 3):
                y, z = nbrs[i], nbrs[j]
                try:
                    d = nx.shortest_path_length(G, y, z)
                except nx.NetworkXNoPath:
                    return None
                sizes.append(d + 2)
    finally:
        # restore x and edges
        G.add_node(x, **attrs)
        for y in nbrs:
            if G.has_node(y):
                G.add_edge(x, y)
    sizes.sort()
    return tuple(sizes)  # e.g. (5,6,8) or (6,6,8)
def _is_on_same_6ring(G, v2, v3, v4):
    """Check (v2,v3,v4) lie on the same 6-ring with edge (v3,v4) present.
    Heuristic: temporarily remove (v3,v4); shortest path length(v2,v4) == 4 → 6-cycle closure.
    """
    if not (G.has_node(v2) and G.has_node(v3) and G.has_node(v4)):
        return False
    if not G.has_edge(v3, v4):
        return False
    # Temporarily remove (v3,v4)
    G.remove_edge(v3, v4)
    ok = False
    try:
        try:
            sp = nx.shortest_path_length(G, v2, v4)
            ok = (sp == 4)
        except nx.NetworkXNoPath:
            ok = False
    finally:
        G.add_edge(v3, v4)
    return ok
def _temp_remove_edge_find_paths_lengths(G, u, v, target_lengths, cutoff):
    if not _edge_exists(G, u, v):
        return {L: False for L in target_lengths}
    G.remove_edge(u, v)
    found = {L: False for L in target_lengths}
    try:
        try:
            for p in nx.all_simple_paths(G, source=u, target=v, cutoff=cutoff):
                L = len(p) - 1
                if L in found and not found[L]:
                    found[L] = True
                    if all(found.values()):
                        break
        except nx.NetworkXNoPath:
            pass
    finally:
        G.add_edge(u, v)
    return found
def dv_A(H, symmetry, axial_map, pivot_ax, rng, processed_nodes):
    """
    Stage A (5-8-5) — ALL OR NOTHING (AON).
    """
    N = _neighbors_cache(H)
    log_for_a = []

    # 1) enumerate pristine a1-a2 candidates (same criteria as before)
    pristine_candidates = []
    for a1, a2 in H.edges():
        if not (H.degree(a1) == 3 and H.degree(a2) == 3 and
                H.nodes[a1].get('elem') == 'C' and H.nodes[a2].get('elem') == 'C'):
            continue
        a1_nbrs = list(set(N[a1]) - {a2})
        a2_nbrs = list(set(N[a2]) - {a1})
        if len(a1_nbrs) != 2 or len(a2_nbrs) != 2:
            continue
        if not set(a1_nbrs).isdisjoint(a2_nbrs):
            continue
        n1, n2 = a1_nbrs
        n3, n4 = a2_nbrs
        try:
            if not (H.degree(n1) == 3 and H.nodes[n1]['elem'] == 'C' and
                    H.degree(n2) == 3 and H.nodes[n2]['elem'] == 'C' and
                    H.degree(n3) == 3 and H.nodes[n3]['elem'] == 'C' and
                    H.degree(n4) == 3 and H.nodes[n4]['elem'] == 'C'):
                continue
        except Exception:
            continue
        pristine_candidates.append((a1, a2, n1, n2, n3, n4))

    if not pristine_candidates:
        return False, H, None, 0, [], set(), ['dv_A_none']

    rng.shuffle(pristine_candidates)

    for a1, a2, n1, n2, n3, n4 in pristine_candidates:
        if a1 in processed_nodes or a2 in processed_nodes:
            continue

        # 2) get orbits on ORIGINAL H
        try:
            atom_set_A = {'a1': a1, 'a2': a2, 'n1': n1, 'n2': n2, 'n3': n3, 'n4': n4}
            orbits_A = {}
            for k, v in atom_set_A.items():
                if not H.has_node(v):
                    raise RuntimeError("seed disappeared before orbit query")
                orbits_A[k] = _get_orbital_atoms(H, v, symmetry, axial_map, pivot_ax)
            L = len(orbits_A['a1'])
            if not all(len(v) == L for v in orbits_A.values()):
                continue
            # fixed mirror
            if symmetry == 'mirror':
                if any(len(orbits_A[key]) == 1 for key in ['a1', 'a2', 'n1', 'n2', 'n3', 'n4']):
                    continue
        except Exception:
            continue

        # 3) prepare tuples for A, and node sets only for bookkeeping/logging
        tuples_A = list(zip(orbits_A['a1'], orbits_A['a2'],
                            orbits_A['n1'], orbits_A['n2'], orbits_A['n3'], orbits_A['n4']))

        orbit_a1_set = frozenset(orbits_A['a1'])
        orbit_a2_set = frozenset(orbits_A['a2'])
        orbit_n_set  = frozenset(orbits_A['n1']) | frozenset(orbits_A['n2']) | \
                       frozenset(orbits_A['n3']) | frozenset(orbits_A['n4'])

        # avoid reusing nodes that caller marked as processed
        if (orbit_a1_set | orbit_a2_set | orbit_n_set) & processed_nodes:
            continue

        # 4) simulate A on a COPY; if any tuple fails -> rollback this candidate
        H2 = H.copy()
        dv_B_atom_sets = []
        failed = False

        for x_a1, x_a2, x_n1, x_n2, x_n3, x_n4 in tuples_A:
            # preconditions must hold on CURRENT H2 (not only H)
            if not (_safe_has_nodes(H2, [x_a1, x_a2, x_n1, x_n2, x_n3, x_n4]) and
                    _edge_exists(H2, x_a1, x_a2)):
                failed = True
                break

            # apply 5-8-5 (remove a1,a2 and add n1-n2 / n3-n4 if missing)
            try:
                H2.remove_node(x_a1)
                H2.remove_node(x_a2)
            except Exception:
                failed = True
                break

            if _safe_has_nodes(H2, [x_n1, x_n2]) and not _edge_exists(H2, x_n1, x_n2):
                H2.add_edge(x_n1, x_n2)
            if _safe_has_nodes(H2, [x_n3, x_n4]) and not _edge_exists(H2, x_n3, x_n4):
                H2.add_edge(x_n3, x_n4)

            # ===== find B-seed for THIS tuple on the UPDATED H2 =====
            x_b1 = x_n1
            x_b2 = x_b3 = x_b4 = None

            # b2: temp-remove edge test → need a 7-step path
            ok = False
            for potential_b2 in list(H2.neighbors(x_b1)):
                if potential_b2 == x_n2:
                    continue
                if _temp_remove_edge_has_path_len(H2, x_b1, potential_b2, need_len=7):
                    x_b2 = potential_b2
                    ok = True
                    break
            if not ok:
                failed = True
                break

            # b3: neighbors of b2 having both 5 and 7 length simple paths to b2
            ok = False
            for potential_b3 in list(H2.neighbors(x_b2)):
                if potential_b3 == x_b1:
                    continue
                found = _temp_remove_edge_find_paths_lengths(
                    H2, x_b2, potential_b3, target_lengths={5, 7}, cutoff=7
                )
                if found.get(5) and found.get(7):
                    x_b3 = potential_b3
                    ok = True
                    break
            if not ok:
                failed = True
                break

            # b4: the "ladder closure" probe around b3
            ok = False
            for potential_b4 in list(H2.neighbors(x_b3)):
                if potential_b4 == x_b2:
                    continue
                hit = False
                for Z in list(H2.neighbors(potential_b4)):
                    if Z == x_b3:
                        continue
                    for Y in list(H2.neighbors(Z)):
                        if Y == potential_b4:
                            continue
                        for X in list(H2.neighbors(Y)):
                            if X == Z:
                                continue
                            if _edge_exists(H2, X, x_b2) and X != x_b3:
                                hit = True
                                break
                        if hit: break
                    if hit: break
                if hit:
                    x_b4 = potential_b4
                    ok = True
                    break
            if not ok:
                failed = True
                break

            dv_B_atom_sets.append((x_b1, x_b2, x_b3, x_b4))

        if failed:
            # rollback this candidate and try next one
            continue

        # success: ALL tuples executed A and produced B-candidates
        if hasattr(H2, "_kdtree_cache"):
            try: delattr(H2, "_kdtree_cache")
            except: pass
        _invalidate_hex_cache(H2)

        delta_proc = set(orbit_a1_set) | set(orbit_a2_set) | set(orbit_n_set)
        return True, H2, orbits_A, L, dv_B_atom_sets, delta_proc, log_for_a

    # no candidate survived
    return False, H, None, 0, [], set(), ['dv_A_none']

def dv_B(H, orbits_A, dv_B_atom_sets, orbit_len, rng, symmetry, axial_map, pivot_ax):
    """
    B: 555-777 (ALL-OR-NOTHING)
    """
    if not orbits_A or orbit_len <= 0:
        return False, H, []
    # For each orbit index, we will build (b1,b2,b3,b4) on the CURRENT graph H.
    # We mostly expect b1 to be from orbit n1 or n3; we will try n1 first, then n3.
    n1_orb = orbits_A['n1']
    n2_orb = orbits_A['n2']
    n3_orb = orbits_A['n3']
    n4_orb = orbits_A['n4']

    tuples_B = []
    # ---- GLOBAL DISCOVERY on ORIGINAL H (AON: all L entries must succeed) ----
    for k in range(orbit_len):
        # prefer n1 as b1 if it satisfies {5,6,8}, else try n3
        b1_candidate_list = [n1_orb[k], n3_orb[k]]

        b1 = b2 = b3 = b4 = None
        for cand_b1 in b1_candidate_list:
            if cand_b1 not in H or H.degree(cand_b1) != 3:
                continue
            sig_b1 = _ring_signature_deg3(H.copy(), cand_b1)
            if sig_b1 != (5, 6, 8):
                continue

            # find UNIQUE neighbor with signature (6,6,8)
            candidates_b2 = []
            for nb in H.neighbors(cand_b1):
                if nb == n2_orb[k] or nb == n4_orb[k]:
                    # optional: skip the "paired" partner of the new A-bond, but not strictly required.
                    pass
                sig_nb = _ring_signature_deg3(H.copy(), nb)
                if sig_nb == (6, 6, 8):
                    candidates_b2.append(nb)

            if len(candidates_b2) != 1:
                # must be unique
                continue
            cand_b2 = candidates_b2[0]

            # b3: neighbor of b2 (not b1) with signature (6,6,8)
            cand_b3 = None
            for nb2 in H.neighbors(cand_b2):
                if nb2 == cand_b1:
                    continue
                sig_nb2 = _ring_signature_deg3(H.copy(), nb2)
                if sig_nb2 == (6, 6, 8):
                    cand_b3 = nb2
                    break
            if cand_b3 is None:
                continue

            # b4: neighbor of b3 (not b2) s.t. (b2,b3,b4) share a 6-ring and (b3,b4) is an edge
            cand_b4 = None
            for nb3 in H.neighbors(cand_b3):
                if nb3 == cand_b2:
                    continue
                if _is_on_same_6ring(H, cand_b2, cand_b3, nb3):
                    cand_b4 = nb3
                    break
            if cand_b4 is None:
                continue

            # success for this orbit index
            b1, b2, b3, b4 = cand_b1, cand_b2, cand_b3, cand_b4
            break

        if not all(v is not None for v in (b1, b2, b3, b4)):
            # any failure → AON abort
            return False, H, []

        tuples_B.append((b1, b2, b3, b4))
    # ---- APPLY to a COPY (AON) ----
    H2 = H.copy()
    try:
        for (b1, b2, b3, b4) in tuples_B:
            if not (H2.has_node(b1) and H2.has_node(b2) and H2.has_node(b3) and H2.has_node(b4)):
                raise RuntimeError("B: node missing in H2")
            if not (H2.has_edge(b1, b2) and H2.has_edge(b3, b4)):
                raise RuntimeError("B: required pre-edges missing")

            # remove (b1-b2) & (b3-b4)
            H2.remove_edge(b1, b2)
            H2.remove_edge(b3, b4)
            # add (b1-b3) & (b2-b4)
            if not H2.has_edge(b1, b3): H2.add_edge(b1, b3)
            if not H2.has_edge(b2, b4): H2.add_edge(b2, b4)

    except Exception:
        return False, H, []

    if hasattr(H2, "_kdtree_cache"):
        try: delattr(H2, "_kdtree_cache")
        except: pass
    _invalidate_hex_cache(H2)

    # Prepare C stage seeds: a1 := b2, a2..a4 will be discovered in dv_C
    dv_C_atom_sets = [(b2, None, None, None) for (b1, b2, b3, b4) in tuples_B]
    return True, H2, dv_C_atom_sets
def dv_C(H, dv_C_atom_sets, orbit_len, rng, symmetry, axial_map, pivot_ax):
    """
    C: 5555-6-7777 (ALL-OR-NOTHING)
    """
    if not dv_C_atom_sets or orbit_len <= 0:
        return False, H
    tuples_C = []
    # ---- GLOBAL DISCOVERY on ORIGINAL H (AON) ----
    # We only stored a1=b2 placeholder from dv_B; here we actually *discover* a1..a4 by ring signatures.
    for k in range(orbit_len):
        a1 = a2 = a3 = a4 = None
        # Find a1 as any degree-3 node with signature (5,6,7) near previous B rewiring.
        # Start search from the previously returned placeholder dv_C_atom_sets[k][0] (which equals b2),
        # then check this node and its local 2-hop neighborhood to speed things up.
        seed = dv_C_atom_sets[k][0]
        frontier = set([seed])
        frontier |= set(H.neighbors(seed))
        for v in list(H.neighbors(seed)):
            frontier |= set(H.neighbors(v))

        # Fallback to whole graph if local search fails
        candidate_space = list(frontier) if frontier else list(H.nodes())

        # Step 1: pick a1 (signature {5,6,7})
        for cand_a1 in candidate_space:
            sig = _ring_signature_deg3(H.copy(), cand_a1)
            if sig == (5, 6, 7):
                a1 = cand_a1
                break
        if a1 is None:
            return False, H

        # Step 2: a2 — UNIQUE neighbor of a1 with signature {7,6,6}
        candidates_a2 = []
        for nb in H.neighbors(a1):
            sig_nb = _ring_signature_deg3(H.copy(), nb)
            if sig_nb == (6, 6, 7):
                candidates_a2.append(nb)
        if len(candidates_a2) != 1:
            return False, H
        a2 = candidates_a2[0]

        # Step 3: a3 — neighbor of a2 (not a1) with signature {7,6,6}
        for nb2 in H.neighbors(a2):
            if nb2 == a1:
                continue
            sig_nb2 = _ring_signature_deg3(H.copy(), nb2)
            if sig_nb2 == (6, 6, 7):
                a3 = nb2
                break
        if a3 is None:
            return False, H

        # Step 4: a4 — neighbor of a3 (not a2) s.t. (a2,a3,a4) share a 6-ring
        for nb3 in H.neighbors(a3):
            if nb3 == a2:
                continue
            if _is_on_same_6ring(H, a2, a3, nb3):
                a4 = nb3
                break
        if a4 is None:
            return False, H

        tuples_C.append((a1, a2, a3, a4))
    # ---- APPLY to a COPY (AON) ----
    H2 = H.copy()
    try:
        for (a1, a2, a3, a4) in tuples_C:
            if not (H2.has_node(a1) and H2.has_node(a2) and H2.has_node(a3) and H2.has_node(a4)):
                raise RuntimeError("C: node missing in H2")
            if not (H2.has_edge(a1, a2) and H2.has_edge(a3, a4)):
                raise RuntimeError("C: required pre-edges missing")

            H2.remove_edge(a1, a2)
            H2.remove_edge(a3, a4)
            if not H2.has_edge(a1, a3): H2.add_edge(a1, a3)
            if not H2.has_edge(a2, a4): H2.add_edge(a2, a4)

    except Exception:
        return False, H

    if hasattr(H2, "_kdtree_cache"):
        try: delattr(H2, "_kdtree_cache")
        except: pass
    _invalidate_hex_cache(H2)

    return True, H2
def apply_symmetric_dv(G, symmetry, axial_map, pivot_ax, num_sites=1, rng=None, log_list=None):
    """
    wrapper function to apply the full DV defect pipeline: A -> B -> C
    """
    rng = _ensure_rng(rng)
    if log_list is None:
        log_list = []

    H = G.copy()
    processed_nodes = set()
    used_sites = 0

    while used_sites < num_sites:
        # A
        okA, H, orbits_A, orbit_len, dv_B_sets, delta_proc, _loga = dv_A(
            H, symmetry, axial_map, pivot_ax, rng, processed_nodes
        )
        if not okA:
            break
        processed_nodes |= delta_proc

        # B
        okB, H, dv_C_sets = dv_B(H, orbits_A, dv_B_sets, orbit_len, rng, symmetry, axial_map, pivot_ax)
        if not okB:
            log_list.append('dv_A')
            used_sites += 1
            continue
        # C
        okC, H = dv_C(H, dv_C_sets, orbit_len, rng, symmetry, axial_map, pivot_ax)
        if okC:
            log_list.append('dv_C')
        else:
            log_list.append('dv_B')

        used_sites += 1

    _invalidate_hex_cache(H)
    return H
def apply_symmetric_edge_defect(G, symmetry, axial_map, pivot_ax, num_sites=1, rng=None):
    """
    This is a RECONSTRUCTIVE defect:
    Adds a new bond between its two neighbors (turning a 6-ring into a 5-ring).
    """
    rng = _ensure_rng(rng)
    H = G.copy()

    # 1. Selection: Find all valid "edge" atoms for 5-ring creation.
    # A valid seed is 'C', degree 2, with two 'C' neighbors that are not already bonded.
    valid_seed_candidates = []
    for node in H.nodes():
        if (H.nodes[node]['elem'] == 'C' and
                H.degree(node) == 2 and
                node in axial_map):

            neighbors = list(H.neighbors(node))
            if len(neighbors) == 2:
                n1, n2 = neighbors

                # Ensure neighbors are Carbon
                if not (H.has_node(n1) and H.has_node(n2) and
                        H.nodes[n1]['elem'] == 'C' and H.nodes[n2]['elem'] == 'C'):
                    continue

                # Ensure neighbors are not already bonded to each other (e.g., in a 3-ring)
                if H.has_edge(n1, n2):
                    continue

                # Check neighbor degrees (allows Armchair 3-2-2 and Zigzag 3-2-3)
                d1 = H.degree(n1)
                d2 = H.degree(n2)

                # 3-2-3 (Zigzag) or (3-2-2 / 2-2-3) (Armchair)
                if (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 2) or (d1 == 2 and d2 == 3):
                    valid_seed_candidates.append((node, n1, n2))

    if not valid_seed_candidates:
        print("No valid edge atoms (deg=2 C, with C neighbors) found for edge defect.")
        return G

    rng.shuffle(valid_seed_candidates)

    # 2. Apply removal/reconstruction orbit by orbit
    removed_nodes = set()
    used = 0

    for (seed_atom, n1, n2) in valid_seed_candidates:
        if used >= num_sites:
            break
        if seed_atom in removed_nodes:
            continue

        # 3. Verify symmetric reconstruction is possible
        try:
            orb_atom_list = _get_orbital_atoms(H, seed_atom, symmetry, axial_map, pivot_ax)
            orb_n1 = _get_orbital_atoms(H, n1, symmetry, axial_map, pivot_ax)
            orb_n2 = _get_orbital_atoms(H, n2, symmetry, axial_map, pivot_ax)
        except Exception:
            continue

        if not (len(orb_atom_list) == len(orb_n1) == len(orb_n2)):
            # print(f"Skipping edge defect at {seed_atom}: Asymmetric orbits.")
            continue  # Asymmetric orbits

        bonds_to_add = list(zip(orb_n1, orb_n2))
        orb_set = frozenset(orb_atom_list)

        # 4. Check for overlaps
        nodes_to_remove_now = orb_set
        atoms_in_new_bonds = {n for pair in bonds_to_add for n in pair}

        if (nodes_to_remove_now.isdisjoint(removed_nodes) and
                atoms_in_new_bonds.isdisjoint(nodes_to_remove_now)):

            # 5. Apply the defect (remove atom, add new bond)
            H.remove_nodes_from(nodes_to_remove_now)
            for u, v in bonds_to_add:
                if H.has_node(u) and H.has_node(v) and not H.has_edge(u, v):
                    H.add_edge(u, v)

            removed_nodes.update(nodes_to_remove_now)
            used += 1

    if used == 0:
        return G  # Nothing
    _invalidate_hex_cache(H)
    return H
def substitute_symmetric_atoms(G, symmetry, axial_map, pivot_ax, elem, num_sites=1, rng=None):
    """
    Substitutes atoms after a strict selection process.
    Selects only C atoms with the correct target degree.
    """
    rng = _ensure_rng(rng)
    degree_map = {'O': 2, 'N': 3, 'B': 3, 'S': 2}
    target_deg = degree_map.get(elem)
    if target_deg is None:
        return G

    H = G.copy()

    # 1. Strict Selection: Find all C atoms that match the target degree.
    valid_seed_candidates = []
    for node in H.nodes():
        if (H.nodes[node]['elem'] == 'C' and
                H.degree(node) == target_deg and
                node in axial_map):
            valid_seed_candidates.append(node)

    if not valid_seed_candidates:
        # print(f"No valid C atoms with degree={target_deg} found for {elem} substitution.")
        return H

    rng.shuffle(valid_seed_candidates)

    # 2. Apply substitution orbit by orbit
    processed_nodes = set()
    used = 0

    for seed_atom in valid_seed_candidates:
        if used >= num_sites:
            break
        if seed_atom in processed_nodes:
            continue  # This atom is already part of a processed orbit

        try:
            orb_set = frozenset(_get_orbital_atoms(H, seed_atom, symmetry, axial_map, pivot_ax))
        except Exception:
            continue  # Failed to get orbit

        # Check for overlap
        if orb_set.isdisjoint(processed_nodes):

            # Final check: Are all atoms in the orbit *still* valid?
            all_valid = True
            for n in orb_set:
                if (n not in H.nodes() or
                        H.nodes[n]['elem'] != 'C' or
                        H.degree(n) != target_deg):
                    all_valid = False
                    break

            if all_valid:
                # 3. Apply the substitution
                for n in orb_set:
                    H.nodes[n]['elem'] = elem
                processed_nodes.update(orb_set)
                used += 1
    return H
# =============================================================================
# SECTION 6: CHEMISTRY UTILS & MAIN WORKER
# =============================================================================
def violates_hetero_rules(mol):
    no_elems = {'N', 'O', 'S'}
    for atom1 in mol.GetAtoms():
        symbol1 = atom1.GetSymbol()
        is_no1 = symbol1 in no_elems
        is_b1 = symbol1 == 'B'

        if not is_no1 and not is_b1:
            continue

        for atom2 in atom1.GetNeighbors():
            symbol2 = atom2.GetSymbol()
            is_no2 = symbol2 in no_elems
            is_b2 = symbol2 == 'B'

            if (symbol1 == 'O' and symbol2 == 'O') or (symbol1 == 'S' and symbol2 == 'S'):
                return True

            if (is_no1 and is_no2):

                for atom3 in atom2.GetNeighbors():
                    if atom3.GetIdx() == atom1.GetIdx():
                        continue

                    symbol3 = atom3.GetSymbol()

                    # (N/O)-(N/O)-(N/O)
                    if is_no1 and is_no2 and (symbol3 in no_elems):
                        return True

            if is_b1 and is_b2:
                return True
    return False
def graph_with_hetero_to_mol(G):
    """
    Creates an RDKit molecule from a NetworkX graph,
    including index correction and chemical sanity checks to prevent library crashes.
    """
    mol = Chem.RWMol()
    valence = {'C': 4, 'N': 3, 'O': 2, 'B': 3, 'S': 2}
    node_to_idx = {}

    # 1. Add atoms and populate mapping.
    for n in G.nodes():
        elem = G.nodes[n].get('elem', 'C')
        if not isinstance(elem, str) or not elem.strip(): elem = 'C'
        elem = elem.strip()[0].upper()
        if elem not in valence: elem = 'C'
        atom = Chem.Atom(elem)
        # atom.SetNoImplicit(True)
        idx = mol.AddAtom(atom)
        node_to_idx[n] = idx

    # 2. Add correct chemical bonds using the mapping.
    for u, v in G.edges():
        idx_u = node_to_idx.get(u)
        idx_v = node_to_idx.get(v)
        if idx_u is not None and idx_v is not None:
            mol.AddBond(idx_u, idx_v, Chem.BondType.SINGLE)

    # 3. Perform chemical sanity check before any operations ***
    try:
        # SanitizeMol checks valence and other chemical rules.
        # It will raise an exception if the graph structure is invalid.
        Chem.SanitizeMol(mol, catchErrors=True)
    except Exception as e:
        # If the check fails, the molecule is problematic. Mark as invalid and skip.
        # print(f"[WARN] RDKit sanitization failed for a molecule: {e}. Skipping.")
        return None

    return mol
def get_structure_from_mol2d(mol_2d):
    mol = assign_kekule_by_matching(mol_2d)
    mol = charge_carbons_without_double(mol)

    smi = Chem.MolToSmiles(mol, canonical=True)
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    try:
        Chem.SanitizeMol(
            mol,
            catchErrors=True,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        )
    except Exception:
        return None, None

    mol = Chem.AddHs(mol)

    try:
        result = AllChem.EmbedMolecule(mol, params=AllChem.ETKDGv3())
    except Exception:
        return None, None

    if result == -1:
        return None, None

    try:
        conformer = mol.GetConformer(0)
    except ValueError:
        return None, None

    coords = np.array(
        [list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=float
    )

    ase_atoms = Atoms(numbers=[a.GetAtomicNum() for a in mol.GetAtoms()], positions=coords)
    return ase_atoms, mol
def assign_kekule_by_matching(mol):
    """
    Set an approximate Kekulé pattern:
    - consider only C(sp2)-C(sp2) bonds
    - choose a maximum matching of atoms; bonds in the matching -> DOUBLE
    - others remain SINGLE
    """
    rw = Chem.RWMol(mol)

    # Build a graph over carbon atoms only, edges for C–C bonds present in mol
    G = nx.Graph()
    carbons = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == 'C']
    G.add_nodes_from(carbons)

    cc_bond_by_pair = {}
    for b in rw.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if a1.GetSymbol() == 'C' and a2.GetSymbol() == 'C':
            i, j = a1.GetIdx(), a2.GetIdx()
            G.add_edge(i, j)
            cc_bond_by_pair[tuple(sorted((i, j)))] = b.GetIdx()

    # Maximum matching: set those bonds to DOUBLE, others to SINGLE
    M = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
    matched_pairs = {tuple(sorted(e)) for e in M}

    # First set all C–C bonds SINGLE, then upgrade matched ones to DOUBLE
    for (i, j), bidx in cc_bond_by_pair.items():
        rw.GetBondWithIdx(bidx).SetBondType(Chem.BondType.SINGLE)
    for (i, j) in matched_pairs:
        bidx = cc_bond_by_pair[(i, j)]
        rw.GetBondWithIdx(bidx).SetBondType(Chem.BondType.DOUBLE)
    mol2 = rw.GetMol()
    return mol2
def charge_carbons_without_double(mol):
    """
    For any carbon with degree=3 and no double bonds,
    set formal charge +1 and prevent implicit Hs.
    """
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        if a.GetSymbol() != 'C':
            continue
        # check double or aromatic bonds
        has_double = any(b.GetBondType() == Chem.BondType.DOUBLE for b in a.GetBonds())
        has_aromatic = any(b.GetBondType() == Chem.BondType.AROMATIC for b in a.GetBonds())
        if not has_double and not has_aromatic:
            a.SetFormalCharge(+1)
            a.SetNoImplicit(False)
    new = rw.GetMol()
    return new

def get_canonical_smiles_hash(mol):
    """Returns the SHA256 of the Canonical SMILES (used for deduplication/naming)."""
    smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    return hashlib.sha256(smi.encode()).hexdigest()[:24]
# Check for atoms that are too close.
def has_too_close_atoms(atoms, dmin=0.8):
    d = atoms.get_all_distances(mic=False)
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] < dmin:
                return True
    return False
def tune_sqlite_db():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA cache_size=-100000;")
        conn.commit()
    finally:
        conn.close()
# =============================================================================
# SECTION 7: MAIN EXECUTION
# =============================================================================
def get_db():
    global DB_CONN
    if DB_CONN is None:
        DB_CONN = connect(DB_PATH)
    return DB_CONN
def run_one(task_id, total_target, seed,
            p_N, p_O, p_B, p_S,
            do_sw, do_sv, do_dv, do_edge,
            max_defect_types, core_trim_p,
            db_locker=None):
    """Wraps the single molecule generation logic to be callable by multiprocessing."""
    try:
        rng = random.Random(seed)
        print(f"\n[Worker {os.getpid()}] --- Starting task {task_id} (Target: {total_target} successes) ---")
        # Define symmetries and their relative probabilities
        symmetries = ['C2', 'C3', 'C6', 'mirror', 'asymmetric']
        symmetry_weights = SYM_WIGHTS
        target_symmetry = rng.choices(symmetries, weights=symmetry_weights, k=1)[0]
        num_rings = rng.randint(MIN_RINGS, MAX_RINGS)
        # print(f"Target symmetry: {target_symmetry}, rings: ~{num_rings}")

        centers, pivot_ax = grow_random_symmetric(steps=num_rings, symmetry=target_symmetry, rng=rng)
        if not centers:
            # print(f"[Task {task_id}] Failed to generate centers. Skipping.")
            return False

        # Symmetric random core trimming with given probability
        if (core_trim_p > 0.0
                and target_symmetry in ('C6', 'C3', 'C2')  # rotational symmetries only
                and rng.random() < core_trim_p
        ):
            centers = symmetric_random_ring_delete(
                centers,
                target_symmetry,
                pivot_ax,
                rng
            )
            if not centers:
                # print(f"[Task {task_id}] Core trimming removed all centers. Skipping.")
                return False

        # infer pivot from centers and intended symmetry (use target_symmetry here)
        pivot_ax = infer_pivot_ax(centers, target_symmetry)
        flake_graph, axial_map = centers_to_graph(centers)
        initial_symmetry = detect_highest_symmetry(
            flake_graph, axial_map, centers=centers, pivot_ax=pivot_ax
        )

        try:
            flake_mol_2d = graph_with_hetero_to_mol(flake_graph)
            flake_mol_2d = cleanup_fragments(flake_mol_2d)
            if flake_mol_2d is None or flake_mol_2d.GetNumAtoms() == 0:
                # print(f"[Task {task_id}] Failed to generate flake RDKit molecule. Skipping.")
                return False
        except Exception as e:
            # print(f"[Task {task_id}] Error generating flake molecule: {e}")
            return False

        hash_prefix = get_canonical_smiles_hash(flake_mol_2d)
        print(f"[Task {task_id}] Initial actual symmetry: {initial_symmetry}. Hash: {hash_prefix}")

        # ---------- Apply Structural Defects ----------
        # 1. Build a pool of *available* defect types based on switches
        def _to_prob(x):
            if isinstance(x, bool):
                return 1.0 if x else 0.0
            try:
                p = float(x)
            except:
                return 0.0
            return max(0.0, min(1.0, p))

        p_sw   = _to_prob(do_sw)
        p_sv   = _to_prob(do_sv)
        p_dv   = _to_prob(do_dv)
        p_edge = _to_prob(do_edge)

        available_defects = []
        if rng.random() < p_sw:
            available_defects.append("sw")
        if rng.random() < p_sv:
            available_defects.append("sv")
        if rng.random() < p_dv:
            available_defects.append("dv")
        if rng.random() < p_edge:
            available_defects.append("edge")

        defective_graph = flake_graph.copy()  # Start with the original flake
        applied_defects_log = []  # To track what we applied for the filename

        if not available_defects or max_defect_types == 0:
            # print(f"[Task {task_id}] No structural defects enabled or max_defect_types=0.")
            defect_type_str = "none"
        else:
            # 2. Decide how many *different types* of defects to apply (0, 1, or up to max)
            # Ensure we don't try to apply more types than are available
            max_types_this_run = min(len(available_defects), max_defect_types)
            num_to_apply = rng.randint(1, max_types_this_run)

            if num_to_apply == 0:
                # print(f"[Task {task_id}] Randomly chose to apply 0 structural defects.")
                defect_type_str = "none"
            else:
                # 3. Sample *which* defects to apply from the pool
                defects_to_apply = rng.sample(available_defects, num_to_apply)
                # 4. Shuffle the order of application (e.g., sometimes SV then SW, sometimes SW then SV)
                rng.shuffle(defects_to_apply)
                # print(f"[Task {task_id}] Applying {num_to_apply} structural defect(s) in order: {defects_to_apply}")

                for defect_type in defects_to_apply:
                    log_this_defect = True  # Flag to control generic logging

                    # IMPORTANT: Apply the next defect to the *result* of the previous one
                    if defect_type == 'sw':
                        defective_graph = apply_symmetric_sw(defective_graph, initial_symmetry, axial_map, pivot_ax=pivot_ax, rng=rng)

                    elif defect_type == 'sv':
                        num_sv_sites = rng.randint(1, 2)
                        defective_graph = apply_symmetric_sv(defective_graph, initial_symmetry, axial_map, pivot_ax=pivot_ax,
                                                             num_sites=num_sv_sites, rng=rng)
                    elif defect_type == 'dv':
                        num_dv_sites = 1
                        defective_graph = apply_symmetric_dv(defective_graph, initial_symmetry, axial_map, pivot_ax=pivot_ax,
                                                             num_sites=num_dv_sites, rng=rng,
                                                             log_list=applied_defects_log)  # Pass the log
                        log_this_defect = False  # The function will log its own specific type

                    elif defect_type == 'edge':
                        num_edge_sites = rng.randint(1, 2)
                        defective_graph = apply_symmetric_edge_defect(defective_graph, initial_symmetry, axial_map, pivot_ax=pivot_ax,
                                                                      num_sites=num_edge_sites, rng=rng)
                    if log_this_defect:
                        applied_defects_log.append(defect_type)

                # Create a sorted, unique string for the filename, e.g., "sv+sw"
                defect_type_str = "+".join(sorted(list(set(applied_defects_log))))
                if not defect_type_str: defect_type_str = "none"  # Safety check

        defective_graph = add_internal_bonds_nx(defective_graph, initial_symmetry, axial_map, pivot_ax, max_dist=2.0)

        # ---------- Apply Chemical Substitutions (after all structural defects) ----------
        final_graph = defective_graph

        # maximum 35%
        carbon_nodes = [n for n, d in final_graph.nodes(data=True) if d.get("elem", "C") == "C"]
        num_C = len(carbon_nodes)

        sym = initial_symmetry.lower() if initial_symmetry else "asymmetric"
        if sym == "c6":
            sym_fac = 6
        elif sym == "c3":
            sym_fac = 3
        elif sym == "c2":
            sym_fac = 2
        elif sym == "mirror":
            sym_fac = 2
        else:
            sym_fac = 1

        max_sub_allowed = max(1, int(0.35 * num_C / sym_fac))

        nN = nO = nB = nS = 0

        def maybe_substitute(elem_symbol: str, p_elem: float):
            nonlocal final_graph, nN, nO, nB, nS
            if p_elem <= 0.0 or max_sub_allowed <= 0:
                return
            if rng.random() >= p_elem:
                return

            # 1 ~ max_sub_allowed
            raw = int(round((rng.random() ** 3) * max_sub_allowed))
            upper = max(1, raw)
            num_sites_this_time = rng.randint(1, upper)

            retry = 0
            while retry < 5:
                temp_graph = substitute_symmetric_atoms(
                    final_graph.copy(), initial_symmetry, axial_map, pivot_ax,
                    elem=elem_symbol, num_sites=num_sites_this_time, rng=rng
                )
                temp_mol = graph_with_hetero_to_mol(temp_graph)
                if temp_mol is not None and not violates_hetero_rules(temp_mol):
                    final_graph = temp_graph
                    if elem_symbol == 'N':
                        nN += num_sites_this_time
                    elif elem_symbol == 'O':
                        nO += num_sites_this_time
                    elif elem_symbol == 'B':
                        nB += num_sites_this_time
                    elif elem_symbol == 'S':
                        nS += num_sites_this_time
                    break
                retry += 1

        maybe_substitute('N', p_N)
        maybe_substitute('O', p_O)
        maybe_substitute('B', p_B)
        maybe_substitute('S', p_S)

        # --- (Rest of the function: 2D mol, 3D embedding, checks, etc.) ---
        final_graph = _remove_dangling(final_graph)
        final_mol_2d = graph_with_hetero_to_mol(final_graph)
        if final_mol_2d is None or final_mol_2d.GetNumAtoms() == 0:
            # print(f"[Task {task_id}] Failed to generate final RDKit molecule. Skipping.")
            return False

        if violates_hetero_rules(final_mol_2d):
            # print(f"[Task {task_id}] Rejected: contains forbidden heteroatom chain (e.g., N-O-N or B-B-B).")
            return False

        final_ase_atoms, final_mol_3d = get_structure_from_mol2d(final_mol_2d)
        if final_ase_atoms is None or final_mol_3d is None:
            # print(f"[Task {task_id}] Skipping this molecule due to 3D embedding failure.")
            return False
        if has_too_close_atoms(final_ase_atoms):
            # print(f"[Task {task_id}] Rejected: too close atoms.")
            return False

        # ---------- Count 4~5~6~7~8 membered rings ----------
        ring_info = final_mol_2d.GetRingInfo()
        atom_rings = ring_info.AtomRings()

        ring_count = {k: 0 for k in range(3, 10)}
        for ring in atom_rings:
            k = len(ring)
            if 3 <= k <= 9:
                ring_count[k] += 1

        if ring_count[3] != 0:
            # print(f"[Task {task_id}] Rejected: contains {ring_count[3]} 3-membered ring(s).")
            return False
        if ring_count[9] != 0:
            # print(f"[Task {task_id}] Rejected: contains {ring_count[9]} 9-membered ring(s).")
            return False

        # compute canonical hash24
        final_smi = Chem.MolToSmiles(final_mol_3d, isomericSmiles=False, canonical=True)
        hash24 = hashlib.sha256(final_smi.encode()).hexdigest()[:24]

        # count heteroatoms
        nN = nB = nO = nS = 0
        for atom in final_mol_3d.GetAtoms():
            sym = atom.GetSymbol()
            if sym == "N":
                nN += 1
            elif sym == "B":
                nB += 1
            elif sym == "O":
                nO += 1
            elif sym == "S":
                nS += 1

        # metadata stored in database
        metadata = {
            "sym": initial_symmetry,
            "defects": defect_type_str,
            "n4": ring_count.get(4, 0),
            "n5": ring_count.get(5, 0),
            "n6": ring_count.get(6, 0),
            "n7": ring_count.get(7, 0),
            "n8": ring_count.get(8, 0),
            "nN": nN,
            "nB": nB,
            "nO": nO,
            "nS": nS,
            "hash24": hash24
        }

        # write to database
        db = get_db()

        with db_locker:
            if db.count(hash24=hash24) > 0:
                print(f"[Task {task_id}] duplicate in DB: {hash24}")
                return False
            db.write(final_ase_atoms, **metadata)

        print(f"[Task {task_id}] saved to DB only: {hash24}")
        return True

    except Exception as e:
        print(f"[Worker {os.getpid()}] Unhandled exception in task {task_id}: {e}")
        traceback.print_exc()
        return False

def _run_one_subprocess_worker(result_queue, db_locker,
                               task_id, total_target, seed,
                               p_N, p_O, p_B, p_S,
                               do_sw, do_sv, do_dv, do_edge,
                               max_defect_types, core_trim_p):
    success = run_one(task_id, total_target, seed,
                      p_N, p_O, p_B, p_S,
                      do_sw, do_sv, do_dv, do_edge,
                      max_defect_types, core_trim_p,
                      db_locker=db_locker)
    try:
        result_queue.put((task_id, success))
    except Exception as e:
        print(f"[Worker {os.getpid()}] Failed to put result for task {task_id}: {e}")

def main():
    print(f"Using {MAX_PARALLEL} processes | "
          f"Target = {N_MOLECULES} | "
          f"Timeout = {TASK_TIMEOUT}s (+{GRACE_PERIOD}s grace)")

    # Main random generator for task seeds
    rng_master = random.Random()
    tune_sqlite_db()
    # Use explicit spawn context for Windows
    ctx = mp.get_context("spawn")
    db_locker = ctx.Lock()

    # active_tasks: task_id -> {process, queue, start_time}
    active_tasks = {}

    successful_count = 0
    submitted_count = 0

    def start_new_task(task_id):
        """
        Launch a new subprocess running run_one().
        """
        nonlocal submitted_count

        seed = rng_master.randrange(1, 2 ** 20)
        result_queue = ctx.Queue()

        p = ctx.Process(
            target=_run_one_subprocess_worker,
            args=(
                result_queue, db_locker,
                task_id, N_MOLECULES, seed,
                PROB_N, PROB_O, PROB_B, PROB_S,
                PROB_SW, PROB_SV, PROB_DV, PROB_EDGE,
                MAX_DEFECT_TYPES, CORE_TRIM_P,
            )
        )
        p.daemon = False
        p.start()

        active_tasks[task_id] = {
            "process": p,
            "queue": result_queue,
            "start_time": time.time()
        }

        submitted_count += 1

    # === Fill initial worker slots ===
    initial_jobs = min(MAX_PARALLEL, N_MOLECULES)
    for i in range(initial_jobs):
        start_new_task(i)

    next_task_id = initial_jobs

    # === Main supervision loop ===
    while active_tasks:
        time.sleep(0.2)  # avoid CPU busy-wait

        now = time.time()
        finished_task_ids = []

        for task_id, info in list(active_tasks.items()):

            p = info["process"]
            q = info["queue"]
            start_time = info["start_time"]

            # --- Case 1: process finished ---
            if not p.is_alive():
                p.join()

                success_flag = False
                try:
                    while not q.empty():
                        _, success_flag = q.get_nowait()
                except Exception:
                    pass

                if success_flag:
                    successful_count += 1
                    print(f"[Main] Task {task_id} SUCCESS "
                          f"({successful_count}/{N_MOLECULES})")
                else:
                    print(f"[Main] Task {task_id} FAILED")

                finished_task_ids.append(task_id)
                continue

            # --- Case 2: still running, check timeout ---
            elapsed = now - start_time
            if elapsed > TASK_TIMEOUT:
                print(f"[Main] Task {task_id} timeout > {TASK_TIMEOUT}s. "
                      f"Waiting {GRACE_PERIOD}s grace...")

                p.join(GRACE_PERIOD)

                if p.is_alive():
                    p.terminate()
                    p.join()
                    finished_task_ids.append(task_id)
                else:
                    success_flag = False
                    try:
                        while not q.empty():
                            _, success_flag = q.get_nowait()
                    except Exception:
                        pass

                    if success_flag:
                        successful_count += 1
                        print(f"[Main] Task {task_id} SUCCESS (during grace) "
                              f"({successful_count}/{N_MOLECULES})")
                    else:
                        print(f"[Main] Task {task_id} FAILED (during grace)")

                    finished_task_ids.append(task_id)

        # === Cleanup finished tasks and refill queue ===
        for tid in finished_task_ids:
            info = active_tasks.pop(tid, None)

            if info:
                proc = info["process"]
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=0.5)

                try:
                    info["queue"].close()
                except Exception:
                    pass

            if successful_count < N_MOLECULES and len(active_tasks) < MAX_PARALLEL:
                start_new_task(next_task_id)
                next_task_id += 1

    # === Cleanup remaining tasks if target reached ===
    if successful_count >= N_MOLECULES and active_tasks:
        print(f"[Main] Target reached. Killing {len(active_tasks)} remaining tasks...")

        for tid, info in active_tasks.items():
            p = info["process"]
            if p.is_alive():
                print(f"[Main] Closing task {tid}, PID={p.pid}")
                p.close()
                p.join(timeout=0.8)

            try:
                info["queue"].close()
            except Exception:
                pass

    print(f"\nFinished: {successful_count}/{N_MOLECULES} "
          f"successful molecules (submitted {submitted_count} tasks).")

if __name__ == "__main__":
    main()
