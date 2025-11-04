import os, random, hashlib, math, itertools, traceback, time, gc
import networkx as nx
import numpy as np
import multiprocessing as mp
import scipy.optimize as opt
from joblib.externals.loky import cpu_count
from matplotlib import pyplot as plt
from rdkit import Chem, RDLogger, rdBase
RDLogger.DisableLog('rdApp.info')
RDLogger.DisableLog('rdApp.warning')
from rdkit.Chem import AllChem, Draw, rdmolops, rdDepictor, rdchem
from ase import Atoms, io
from ase.visualize import view
from scipy.spatial import KDTree

# =============================================================================
# PART 1: SYMMETRIC FLAKE GENERATION
# Core function: Generate a set of hexagonal center points with specific
# point group symmetry based on an axial coordinate system.
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


def get_symmetric_equivalents_axial(pos, symmetry):
    """
    Return axial (q,r) orbit for the given symmetry with a CONSISTENT order:
    C6: 0, +60, +120, +180, +240, +300
    C3: 0, +120, +240
    C2: 0, +180
    mirror: identity, mirror
    C1/asymmetric: identity
    """
    def axial_to_cube(q, r):
        # pointy-top axial: q, r; cube: (x, y, z) with x + y + z = 0
        x = q
        z = r
        y = -x - z
        return (x, y, z)

    def cube_to_axial(x, y, z):
        return (x, z)

    def rot60_cube(x, y, z):
        # +60° rotation on cube coords: (x,y,z) -> (-z, -x, -y)
        return (-z, -x, -y)

    def rot_k_times(q, r, k):
        x, y, z = axial_to_cube(q, r)
        for _ in range(k):
            x, y, z = rot60_cube(x, y, z)
        return cube_to_axial(x, y, z)

    q, r = pos

    if symmetry == 'C6':
        return [rot_k_times(q, r, k) for k in range(6)]
    elif symmetry == 'C3':
        return [rot_k_times(q, r, k) for k in (0, 2, 4)]
    elif symmetry == 'C2':
        return [rot_k_times(q, r, k) for k in (0, 3)]
    elif symmetry == 'mirror':
        # mirror across the vertical axis in axial (pointy-top); here we mirror (q,r)->(-q-r, r)
        return [(q, r), (-q - r, r)]
    else:  # 'C1' or 'asymmetric'
        return [(q, r)]


def rot2d_deg(v, deg):
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1])

def mirror2d_vertical(v):
    return (-v[0], v[1])

TRANSFORM_OPS = {
    'C6': [lambda v,ang=ang: rot2d_deg(v, ang) for ang in (0, 60, 120, 180, 240, 300)],
    'C3': [lambda v,ang=ang: rot2d_deg(v, ang) for ang in (0, 120, 240)],
    'C2': [lambda v,ang=ang: rot2d_deg(v, ang) for ang in (0, 180)],
    'mirror': [lambda v: v, mirror2d_vertical],
    'asymmetric': [lambda v: v],
    'C1': [lambda v: v],
}

def grow_random_symmetric(steps, symmetry, seed_centers=None, rng=None):
    """
    Performs random growth according to the specified symmetry.
    """
    if rng is None:
        rng = random.Random()
    if seed_centers is None:
        seed_centers = [(0, 0)]  # Symmetric growth must start from the center.

    centers = set()
    # The initial points must also conform to the symmetry.
    initial_equivalents = get_symmetric_equivalents_axial(seed_centers[0], symmetry)
    for p in initial_equivalents:
        centers.add(p)

    nbr_dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    # Loop until the target number of rings is reached.
    while len(centers) < steps + 1:
        # 1. Find all possible growth points (candidates).
        candidates = set()
        for c in centers:
            for d in nbr_dirs:
                pos = (c[0] + d[0], c[1] + d[1])
                if pos not in centers:
                    candidates.add(pos)

        if not candidates:
            break

        # 2. Randomize the order of candidates to try.
        candidate_list = list(candidates)
        rng.shuffle(candidate_list)

        added_successfully = False
        for new in candidate_list:
            # Get all symmetric equivalents for the current candidate.
            new_points = get_symmetric_equivalents_axial(new, symmetry)
            new_points_set = set(new_points)

            # --- Check if all new points are adjacent to the existing structure ---
            all_connected = True
            for p in new_points_set:
                # Check if point p has at least one neighbor in the 'centers' set.
                is_p_connected = False
                for d in nbr_dirs:
                    neighbor = (p[0] + d[0], p[1] + d[1])
                    # Check if the neighbor is in the existing structure OR in the set of new points to be added.
                    if neighbor in centers:
                        is_p_connected = True
                        break

                # If this point cannot find any connection to the existing structure, this set of points is invalid.
                if not is_p_connected:
                    all_connected = False
                    break

            # 3. If all points are validated, add them and proceed to the next growth iteration.
            if all_connected:
                for p in new_points_set:
                    centers.add(p)
                added_successfully = True
                break  # Successfully added, break from the candidate loop.

        # If no valid growth (ensuring connectivity) is found after checking all candidates, stop.
        if not added_successfully:
            print("Warning: Could not find a valid connected growth point. Stopping.")
            break

    return sorted(list(centers))


def add_internal_bonds_nx(G, symmetry, axial_map, max_dist=1.9):
    """
    NetworkX version of add_internal_bonds. Closes internal rings by
    adding new bonds between nodes with degree 2, in a SYMMETRIC way.
    """
    H = G.copy()  # Operate on a copy.
    pos = {n: H.nodes[n]['xy'] for n in H.nodes()}

    while True:
        bonds_added_in_this_iteration = False

        # Find all nodes with degree 2.
        degree2_atoms_indices = [n for n, deg in H.degree() if deg == 2]
        if len(degree2_atoms_indices) < 2:
            break

        potential_bonds = []
        for i, j in itertools.combinations(degree2_atoms_indices, 2):
            # We only need to check seeds where i, j are valid (degree 2)
            if not H.has_edge(i, j) and H.degree(i) == 2 and H.degree(j) == 2:
                (x1, y1) = pos[i]
                (x2, y2) = pos[j]
                dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                if dist < max_dist:
                    try:
                        path_len = nx.shortest_path_length(H, source=i, target=j)
                        new_ring_size = path_len
                        potential_bonds.append({'atoms': (i, j), 'dist': dist, 'ring_size': new_ring_size})
                    except nx.NetworkXNoPath:
                        continue

        if not potential_bonds:
            break

        # Prioritize forming 6-membered rings, then 5-membered, and finally by distance.
        def sort_key(bond_info):
            size = bond_info['ring_size']
            dist = bond_info['dist']
            priority = 2
            if size == 5:
                priority = 0  # Highest priority
            elif size == 4:
                priority = 1
            return (priority, dist)

        potential_bonds.sort(key=sort_key)

        # Find the highest-priority *seed* bond whose *entire orbit* is valid
        for bond_info in potential_bonds:
            seed_i, seed_j = bond_info['atoms']

            # 1. Check if this seed is still valid (nodes might have been
            #    used by a previous, higher-priority orbit in this loop,
            #    although we break, this is a good safety check)
            if H.degree(seed_i) != 2 or H.degree(seed_j) != 2:
                continue

            # 2. Find the symmetric orbit of the *bond*
            try:
                orb_i = _get_orbital_atoms(H, seed_i, symmetry, axial_map)
                orb_j = _get_orbital_atoms(H, seed_j, symmetry, axial_map)
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

def detect_highest_symmetry(G, axial_map):
    """Detects the actual highest symmetry of the graph based on node coordinates + elements."""
    nodes = list(G.nodes())
    if not nodes:
        return 'asymmetric'

    # Build a set of (axial_coordinate, element).
    node_sig = set((axial_map[n], G.nodes[n]['elem']) for n in nodes if n in axial_map)

    def check_symmetry(sym):
        for (q, r), elem in node_sig:
            for eq_q, eq_r in get_symmetric_equivalents_axial((q, r), sym):
                if ((eq_q, eq_r), elem) not in node_sig:
                    return False
        return True

    if check_symmetry('C6'):
        return 'C6'
    if check_symmetry('C3'):
        return 'C3'
    if check_symmetry('C2'):
        return 'C2'
    if check_symmetry('mirror'):
        return 'mirror'
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


# =============================================================================
# PART 2: DEFECT APPLICATION
# Core function: Apply Stone-Wales transformations and atom substitutions
# on the NetworkX graph.
# =============================================================================

def _find_closest_node(G, target_coord, tolerance=5e-2):
    """
    Fast KDTree-based nearest node search.
    Builds a KDTree from node coordinates once per graph.
    """
    if not hasattr(G, "_kdtree_cache"):
        coords = np.array([G.nodes[n]['xy'] for n in G.nodes()])
        G._kdtree_cache = {
            "tree": KDTree(coords),
            "nodes": list(G.nodes())
        }

    tree = G._kdtree_cache["tree"]
    nodes = G._kdtree_cache["nodes"]

    dist, idx = tree.query(target_coord)
    if dist < tolerance:
        return nodes[idx]
    else:
        # only warn occasionally to reduce log spam
        if random.random() < 0.01:
            print(f"[WARN] KDTree: nearest node {dist:.4f} Å away exceeds tolerance {tolerance}.")
        return None

def _get_orbital_atoms(G, ref_node, symmetry, axial_map):
    """
    Returns all atom node IDs that are symmetrically equivalent to ref_node
    (based on axial coordinates).
    """
    centers_list = G.nodes[ref_node].get('ref_centers', [])
    if not centers_list:
        if ref_node not in axial_map:
            return [ref_node]
        centers_list = [axial_map[ref_node]]

    (x, y) = G.nodes[ref_node]['xy']
    vector_ops = TRANSFORM_OPS.get(symmetry, [lambda v: v])

    best_orbital = []
    best_hits = -1

    for (q, r) in centers_list:
        cx, cy = axial_to_cartesian(q, r)
        rel_vec = (x - cx, y - cy)
        centers_axial_orbit = get_symmetric_equivalents_axial((q, r), symmetry)
        if len(centers_axial_orbit) != len(vector_ops):
            continue

        orbital = []
        hits = 0
        for i, center_axial_new in enumerate(centers_axial_orbit):
            cx_new, cy_new = axial_to_cartesian(center_axial_new[0], center_axial_new[1])
            rel_new = vector_ops[i](rel_vec)
            target_coord = (cx_new + rel_new[0], cy_new + rel_new[1])
            node_id = _find_closest_node(G, target_coord)
            if node_id is not None and node_id not in orbital:
                orbital.append(node_id)
                hits += 1

        if hits > best_hits:
            best_hits = hits
            best_orbital = orbital

    return best_orbital if best_orbital else [ref_node]


def _invalidate_hex_cache(G):
    """Invalidates cached 6-ring list stored on the graph object."""
    try:
        if '_hex_cache' in G.graph:
            del G.graph['_hex_cache']
    except Exception:
        pass

def _hexes_cached(G):
    """
    Return 6-cycles using a simple cache on G.graph.
    We keep original _hexes(G)
    """
    cache = G.graph.get('_hex_cache')
    if cache is not None:
        # return a copy to avoid accidental mutation outside
        return [list(c) for c in cache['hexes']]

    # build & store
    try:
        rings6 = [tuple(c) for c in nx.cycle_basis(G) if len(c) == 6]
    except nx.exception.NetworkXNoCycle:
        rings6 = []
    G.graph['_hex_cache'] = {'hexes': rings6}
    return [list(c) for c in rings6]

def _hexes(G):
    """Find all 6-membered rings in the graph."""
    try:
        return [c for c in nx.cycle_basis(G) if len(c) == 6]
    except nx.exception.NetworkXNoCycle:
        return []

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

def apply_symmetric_sw(G, symmetry, axial_map, rng=None):
    all_hexes = _hexes(G)
    if rng is None:
        rng = random.Random()
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
    c1_orb = _get_orbital_atoms(G, c1, symmetry, axial_map)
    c2_orb = _get_orbital_atoms(G, c2, symmetry, axial_map)
    c3_orb = _get_orbital_atoms(G, c3, symmetry, axial_map)
    c4_orb = _get_orbital_atoms(G, c4, symmetry, axial_map)
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


def apply_symmetric_sv(G, symmetry, axial_map, num_sites=1, rng=None):
    """
    Applies Single Vacancy (SV) defects after a strict selection process.
    1. Finds 'C' atoms with degree 3, whose 3 neighbors are also 'C' and degree 3.
    2. Verifies a symmetric reconstruction pairing exists *before* applying.
    """
    if rng is None:
        rng = random.Random()

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
                orb_atom_list = _get_orbital_atoms(H, seed_atom, symmetry, axial_map)
                orb_a = _get_orbital_atoms(H, seed_a, symmetry, axial_map)
                orb_b = _get_orbital_atoms(H, seed_b, symmetry, axial_map)
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
# ------------ helpers: local caches & guards (no logic change) ------------
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

def dv_A(H, symmetry, axial_map, rng, processed_nodes):
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
                orbits_A[k] = _get_orbital_atoms(H, v, symmetry, axial_map)
            L = len(orbits_A['a1'])
            if not all(len(v) == L for v in orbits_A.values()):
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

def dv_B(H, orbits_A, dv_B_atom_sets, orbit_len, rng, symmetry, axial_map):
    """
    B: 555-777 (ALL-OR-NOTHING)
    """

    def __ring_signature_deg3(G, x, cutoff=10):
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
            G.add_node(x, **H.nodes[x])
            for y in nbrs:
                if G.has_node(y):
                    G.add_edge(x, y)
        sizes.sort()
        return tuple(sizes)  # e.g. (5,6,8) or (6,6,8)

    def __is_on_same_6ring(G, b2, b3, b4):
        """Check (b2,b3,b4) lie on the same 6-ring with edge (b3,b4) present.
        Heuristic: temporarily remove (b3,b4); shortest path length(b2,b4) == 4 → 6-cycle closure.
        """
        if not (G.has_node(b2) and G.has_node(b3) and G.has_node(b4)):
            return False
        if not G.has_edge(b3, b4):
            return False
        # Temporarily remove (b3,b4)
        G.remove_edge(b3, b4)
        ok = False
        try:
            try:
                sp = nx.shortest_path_length(G, b2, b4)
                ok = (sp == 4)
            except nx.NetworkXNoPath:
                ok = False
        finally:
            G.add_edge(b3, b4)
        return ok

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
            sig_b1 = __ring_signature_deg3(H.copy(), cand_b1)
            if sig_b1 != (5, 6, 8):
                continue

            # find UNIQUE neighbor with signature (6,6,8)
            candidates_b2 = []
            for nb in H.neighbors(cand_b1):
                if nb == n2_orb[k] or nb == n4_orb[k]:
                    # optional: skip the "paired" partner of the new A-bond, but not strictly required.
                    pass
                sig_nb = __ring_signature_deg3(H.copy(), nb)
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
                sig_nb2 = __ring_signature_deg3(H.copy(), nb2)
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
                if __is_on_same_6ring(H, cand_b2, cand_b3, nb3):
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

def dv_C(H, dv_C_atom_sets, orbit_len, rng, symmetry, axial_map):
    """
    C: 5555-6-7777 (ALL-OR-NOTHING)
    """

    def __ring_signature_deg3(G, x, cutoff=10):
        if x not in G or G.degree(x) != 3:
            return None
        nbrs = list(G.neighbors(x))
        if len(nbrs) != 3:
            return None
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
            G.add_node(x, **H.nodes[x])
            for y in nbrs:
                if G.has_node(y):
                    G.add_edge(x, y)
        sizes.sort()
        return tuple(sizes)  # e.g. (5,6,7) or (6,6,7)

    def __is_on_same_6ring(G, a2, a3, a4):
        if not (G.has_node(a2) and G.has_node(a3) and G.has_node(a4)):
            return False
        if not G.has_edge(a3, a4):
            return False
        G.remove_edge(a3, a4)
        ok = False
        try:
            try:
                sp = nx.shortest_path_length(G, a2, a4)
                ok = (sp == 4)
            except nx.NetworkXNoPath:
                ok = False
        finally:
            G.add_edge(a3, a4)
        return ok

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
            sig = __ring_signature_deg3(H.copy(), cand_a1)
            if sig == (5, 6, 7):
                a1 = cand_a1
                break
        if a1 is None:
            return False, H

        # Step 2: a2 — UNIQUE neighbor of a1 with signature {7,6,6}
        candidates_a2 = []
        for nb in H.neighbors(a1):
            sig_nb = __ring_signature_deg3(H.copy(), nb)
            if sig_nb == (6, 6, 7):
                candidates_a2.append(nb)
        if len(candidates_a2) != 1:
            return False, H
        a2 = candidates_a2[0]

        # Step 3: a3 — neighbor of a2 (not a1) with signature {7,6,6}
        for nb2 in H.neighbors(a2):
            if nb2 == a1:
                continue
            sig_nb2 = __ring_signature_deg3(H.copy(), nb2)
            if sig_nb2 == (6, 6, 7):
                a3 = nb2
                break
        if a3 is None:
            return False, H

        # Step 4: a4 — neighbor of a3 (not a2) s.t. (a2,a3,a4) share a 6-ring
        for nb3 in H.neighbors(a3):
            if nb3 == a2:
                continue
            if __is_on_same_6ring(H, a2, a3, nb3):
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

def apply_symmetric_dv(G, symmetry, axial_map, num_sites=1, rng=None, log_list=None):
    """
    wrapper function to apply the full DV defect pipeline: A -> B -> C
    """
    if rng is None:
        rng = random.Random()
    if log_list is None:
        log_list = []

    H = G.copy()
    processed_nodes = set()
    used_sites = 0

    while used_sites < num_sites:
        # A
        okA, H, orbits_A, orbit_len, dv_B_sets, delta_proc, _loga = dv_A(
            H, symmetry, axial_map, rng, processed_nodes
        )
        if not okA:
            break
        processed_nodes |= delta_proc

        # B
        okB, H, dv_C_sets = dv_B(H, orbits_A, dv_B_sets, orbit_len, rng, symmetry, axial_map)
        if not okB:
            log_list.append('dv_A')
            used_sites += 1
            continue

        # C
        okC, H = dv_C(H, dv_C_sets, orbit_len, rng, symmetry, axial_map)
        if okC:
            log_list.append('dv_C')
        else:
            log_list.append('dv_B')

        used_sites += 1

    _invalidate_hex_cache(H)
    return H


def apply_symmetric_edge_defect(G, symmetry, axial_map, num_sites=1, rng=None):
    """
    This is a RECONSTRUCTIVE defect:
    Adds a new bond between its two neighbors (turning a 6-ring into a 5-ring).
    """
    if rng is None:
        rng = random.Random()

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
            orb_atom_list = _get_orbital_atoms(H, seed_atom, symmetry, axial_map)
            orb_n1 = _get_orbital_atoms(H, n1, symmetry, axial_map)
            orb_n2 = _get_orbital_atoms(H, n2, symmetry, axial_map)
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
        return G  # Nothing was done

    _invalidate_hex_cache(H)
    return H

def substitute_symmetric_atoms(G, symmetry, axial_map, elem, num_sites=1, rng=None):
    """
    Substitutes atoms after a strict selection process.
    Selects only C atoms with the correct target degree.
    """
    if rng is None:
        rng = random.Random()

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
            orb_set = frozenset(_get_orbital_atoms(H, seed_atom, symmetry, axial_map))
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
# PART 3: GRAPH TO MOLECULE CONVERSION
# Core function: Convert a NetworkX graph (with heteroatoms) to an RDKit
# molecule and perform post-processing.
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

# =============================================================================
# PART 4: UTILITIES & EXPORT
# Core function: Helper functions for display, saving, hash calculation, etc.
# =============================================================================

def get_canonical_smiles_hash(mol):
    """Returns the SHA256 of the Canonical SMILES (used for deduplication/naming)."""
    smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    return hashlib.sha256(smi.encode()).hexdigest()[:16]

# Check for atoms that are too close.
def has_too_close_atoms(atoms, dmin=0.8):
    d = atoms.get_all_distances(mic=False)
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] < dmin:
                return True
    return False

def save_mol_file(mol, path):
    """Saves an RDKit molecule as an SD V2000 .mol file."""
    try:
        Chem.MolToMolFile(mol, path)
    except Exception as e:
        print(f"[WARN] Failed to save .mol file to {path}: {e}")

def save_graph_xyz(G, path, comment=""):
    """Saves a NetworkX graph with 2D coordinates as an XYZ file."""
    with open(path, 'w') as f:
        f.write(f"{G.number_of_nodes()}\n")
        f.write(f"{comment}\n")
        for n in sorted(G.nodes()):
            elem = G.nodes[n].get('elem', 'C')
            x, y = G.nodes[n]['xy']
            f.write(f"{elem} {x: .6f} {y: .6f} {0.0: .6f}\n")

def save_mol_svg(mol, path, size=(400, 400)):
    """Saves an RDKit molecule as an SVG."""
    # Generate 2D coordinates for the molecule.
    mol_copy = Chem.Mol(mol)
    Draw.MolToFile(mol_copy, path, size=size)

def save_ase_xyz(atoms, path, comment=""):
    """Saves an ASE Atoms object as an XYZ file."""
    with open(path, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for sym, pos in zip(atoms.get_chemical_symbols(), atoms.positions):
            f.write(f"{sym} {pos[0]: .6f} {pos[1]: .6f} {pos[2]: .6f}\n")

# =============================================================================
# PART 5: MAIN WORKFLOW
# =============================================================================

def run_one(task_id, total_target, seed,
            N_SITES, O_SITES, B_SITES, S_SITES,
            do_N=True, do_O=True, do_B=True, do_S=True,
            do_sw=True, do_sv=True, do_dv=True, do_edge=True,
            max_defect_types=4):
    """Wraps the single molecule generation logic to be callable by multiprocessing."""
    try:
        rng = random.Random(seed)
        print(f"\n[Worker {os.getpid()}] --- Starting task {task_id} (Target: {total_target} successes) ---")
        symmetries = ['C2', 'C3', 'C6', 'mirror', 'asymmetric']
        target_symmetry = rng.choice(symmetries)
        num_rings = rng.randint(10, 120)
        print(f"Target symmetry: {target_symmetry}, rings: ~{num_rings}")

        centers = grow_random_symmetric(steps=num_rings, symmetry=target_symmetry, rng=rng)
        if not centers:
            print(f"[Task {task_id}] Failed to generate centers. Skipping.")
            return False

        flake_graph, axial_map = centers_to_graph(centers)
        initial_symmetry = detect_highest_symmetry(flake_graph, axial_map)

        try:
            flake_mol_2d = graph_with_hetero_to_mol(flake_graph)
            if flake_mol_2d is None or flake_mol_2d.GetNumAtoms() == 0:
                print(f"[Task {task_id}] Failed to generate flake RDKit molecule. Skipping.")
                return False
        except Exception as e:
            print(f"[Task {task_id}] Error generating flake molecule: {e}")
            return False

        hash_prefix = get_canonical_smiles_hash(flake_mol_2d)
        print(f"[Task {task_id}] Initial actual symmetry: {initial_symmetry}. Hash: {hash_prefix}")

        out_path = os.path.join("symmetric_molecules", initial_symmetry)

        # ---------- Apply Structural Defects (MODIFIED for Combination) ----------

        # 1. Build a pool of *available* defect types based on switches
        available_defects = []
        if do_sw: available_defects.append('sw')
        if do_sv: available_defects.append('sv')
        if do_dv: available_defects.append('dv')
        if do_edge: available_defects.append('edge')

        defective_graph = flake_graph.copy()  # Start with the original flake
        applied_defects_log = []  # To track what we applied for the filename

        if not available_defects or max_defect_types == 0:
            print(f"[Task {task_id}] No structural defects enabled or max_defect_types=0.")
            defect_type_str = "none"
        else:
            # 2. Decide how many *different types* of defects to apply (0, 1, or up to max)
            # Ensure we don't try to apply more types than are available
            max_types_this_run = min(len(available_defects), max_defect_types)
            num_to_apply = rng.randint(1, max_types_this_run)

            if num_to_apply == 0:
                print(f"[Task {task_id}] Randomly chose to apply 0 structural defects.")
                defect_type_str = "none"
            else:
                # 3. Sample *which* defects to apply from the pool
                defects_to_apply = rng.sample(available_defects, num_to_apply)
                # 4. Shuffle the order of application (e.g., sometimes SV then SW, sometimes SW then SV)
                rng.shuffle(defects_to_apply)
                print(f"[Task {task_id}] Applying {num_to_apply} structural defect(s) in order: {defects_to_apply}")

                for defect_type in defects_to_apply:
                    log_this_defect = True  # Flag to control generic logging

                    # IMPORTANT: Apply the next defect to the *result* of the previous one
                    if defect_type == 'sw':
                        defective_graph = apply_symmetric_sw(defective_graph, initial_symmetry, axial_map, rng=rng)
                    elif defect_type == 'sv':
                        num_sv_sites = rng.randint(1, 2)
                        defective_graph = apply_symmetric_sv(defective_graph, initial_symmetry, axial_map,
                                                             num_sites=num_sv_sites, rng=rng)
                    elif defect_type == 'dv':
                        num_dv_sites = 1
                        defective_graph = apply_symmetric_dv(defective_graph, initial_symmetry, axial_map,
                                                             num_sites=num_dv_sites, rng=rng,
                                                             log_list=applied_defects_log)  # Pass the log
                        log_this_defect = False  # The function will log its own specific type

                    elif defect_type == 'edge':
                        num_edge_sites = rng.randint(1, 2)
                        defective_graph = apply_symmetric_edge_defect(defective_graph, initial_symmetry, axial_map,
                                                                      num_sites=num_edge_sites, rng=rng)

                    #   if log_this_defect:
                        #   applied_defects_log.append(defect_type)

                # Create a sorted, unique string for the filename, e.g., "sv+sw"
                defect_type_str = "+".join(sorted(list(set(applied_defects_log))))
                if not defect_type_str: defect_type_str = "none"  # Safety check

        defective_graph = add_internal_bonds_nx(defective_graph, initial_symmetry, axial_map, max_dist=1.9)

        # ---------- Apply Chemical Substitutions (after all structural defects) ----------
        final_graph = defective_graph  # This is now the (potentially) multi-defective graph
        if do_N:
            retry = 0
            while retry < 5:
                temp_graph = substitute_symmetric_atoms(
                    final_graph.copy(), initial_symmetry, axial_map,
                    elem='N', num_sites=N_SITES, rng=rng
                )

                temp_mol = graph_with_hetero_to_mol(temp_graph)

                if temp_mol is None:
                    retry += 1
                    continue

                if not violates_hetero_rules(temp_mol):
                    final_graph = temp_graph
                    break
                # ❌ violates → retry
                retry += 1
        if do_O:
            retry = 0
            while retry < 5:
                temp_graph = substitute_symmetric_atoms(
                    final_graph.copy(), initial_symmetry, axial_map,
                    elem='O', num_sites=O_SITES, rng=rng
                )

                temp_mol = graph_with_hetero_to_mol(temp_graph)
                if temp_mol is None:
                    retry += 1
                    continue
                if not violates_hetero_rules(temp_mol):
                    final_graph = temp_graph
                    break
                retry += 1
        if do_B:
            retry = 0
            while retry < 5:
                temp_graph = substitute_symmetric_atoms(
                    final_graph.copy(), initial_symmetry, axial_map,
                    elem='B', num_sites=B_SITES, rng=rng
                )

                temp_mol = graph_with_hetero_to_mol(temp_graph)
                if temp_mol is None:
                    retry += 1
                    continue
                if not violates_hetero_rules(temp_mol):
                    final_graph = temp_graph
                    break
                retry += 1
        if do_S:
            retry = 0
            while retry < 5:
                temp_graph = substitute_symmetric_atoms(
                    final_graph.copy(), initial_symmetry, axial_map,
                    elem='S', num_sites=S_SITES, rng=rng
                )
                temp_mol = graph_with_hetero_to_mol(temp_graph)
                if temp_mol is None:
                    retry += 1
                    continue
                if not violates_hetero_rules(temp_mol):
                    final_graph = temp_graph
                    break
                retry += 1

        # --- (Rest of the function: 2D mol, 3D embedding, checks, etc.) ---
        final_graph = _remove_dangling(final_graph)
        final_mol_2d = graph_with_hetero_to_mol(final_graph)
        if final_mol_2d is None or final_mol_2d.GetNumAtoms() == 0:
            print(f"[Task {task_id}] Failed to generate final RDKit molecule. Skipping.")
            return False

        if violates_hetero_rules(final_mol_2d):
            print(f"[Task {task_id}] Rejected: contains forbidden heteroatom chain (e.g., N-O-N or B-B-B).")
            return False

        final_ase_atoms, final_mol_3d = get_structure_from_mol2d(final_mol_2d)
        if final_ase_atoms is None or final_mol_3d is None:
            print(f"[Task {task_id}] Skipping this molecule due to 3D embedding failure.")
            return False
        if has_too_close_atoms(final_ase_atoms):
            print(f"[Task {task_id}] Rejected: too close atoms.")
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
            print(f"[Task {task_id}] Rejected: contains {ring_count[3]} 3-membered ring(s).")
            return False
        if ring_count[9] != 0:
            print(f"[Task {task_id}] Rejected: contains {ring_count[9]} 9-membered ring(s).")
            return False

        # --- (Modified name tag) ---
        # Use the defect_type_str (e.g., "sv+sw" or "dv" or "none")
        name_tag = "_".join(f"{ring_count[k]}" for k in range(4, 9))
        flake_hash16 = get_canonical_smiles_hash(flake_mol_2d)[:16]
        base_name = f"{name_tag}_{flake_hash16}"

        # ---------- Save final 3D outputs ----------
        out_path = os.path.join("symmetric_molecules", initial_symmetry)
        os.makedirs(out_path, exist_ok=True)

        save_mol_file(final_mol_3d, os.path.join(out_path, f"{base_name}_final.mol"))
        save_ase_xyz(final_ase_atoms, os.path.join(out_path, f"{base_name}_final.xyz"),
                     comment=f"Task {task_id} final")

        final_smi = Chem.MolToSmiles(final_mol_3d, isomericSmiles=False, canonical=True)
        print(f"✅ [Task {task_id}] Successfully saved {base_name} SMILES: {final_smi}")
        return True

    except Exception as e:
        print(f"[Worker {os.getpid()}] Unhandled exception in task {task_id}: {e}")
        traceback.print_exc()
        return False
    gc.collect()


def main():
    N_MOLECULES = 30  # This is the target number of successes.

    # --- Structural Defect Switches (Enable/Disable) ---
    SW_SWITCH = 0  # Enable Stone-Wales
    SV_SWITCH = 0  # Enable Single Vacancy
    DV_SWITCH = 1  # Enable Double Vacancy
    EDGE_SWITCH = 0  # Enable Edge Defect (SV at edge)

    # --- Defect Combination Control ---
    MAX_DEFECT_TYPES_TO_APPLY = 4  # Apply different types

    # --- Chemical Substitution Switches ---
    N_SWITCH = 1
    O_SWITCH = 0
    B_SWITCH = 1
    S_SWITCH = 0

    N_SITES = 2
    O_SITES = 0
    B_SITES = 2
    S_SITES = 2

    TASK_TIMEOUT = 600  # Timeout for a single task (seconds).
    pool_size = cpu_count()-1
    print(f"Using pool size: {pool_size}. Target: {N_MOLECULES} successes. Timeout: {TASK_TIMEOUT}s")

    rng_master = random.Random()
    # Dictionary to track running tasks: {AsyncResult: (task_id, start_time)}
    pending_tasks = {}
    successful_count = 0
    submitted_count = 0

    for sym in ['C6', 'C3', 'C2', 'mirror', 'asymmetric']:
        os.makedirs(os.path.join("symmetric_molecules", sym), exist_ok=True)

    # maxtasksperchild=2 ensures a worker is replaced after two tasks, preventing memory leaks or hangs.
    with mp.Pool(processes=pool_size, maxtasksperchild=2) as pool:

        # Helper function to submit a new task.
        def submit_task(idx):
            seed = rng_master.randrange(1, 2 ** 20)
            args = (idx, N_MOLECULES, seed)
            res = pool.apply_async(run_one, args=(idx, N_MOLECULES, seed,
                                                  N_SITES, O_SITES, B_SITES, S_SITES,
                                                  bool(N_SWITCH), bool(O_SWITCH), bool(B_SWITCH), bool(S_SWITCH),
                                                  bool(SW_SWITCH), bool(SV_SWITCH), bool(DV_SWITCH), bool(EDGE_SWITCH),
                                                  MAX_DEFECT_TYPES_TO_APPLY))
            pending_tasks[res] = (idx, time.time())
            print(f"[Main] Submitted task {idx} (seed {seed})")

        # 1. Start initial tasks (fill the pool).
        for i in range(pool_size):
            submit_task(submitted_count)
            submitted_count += 1

        # 2. Loop to process results until the target is met.
        while successful_count < N_MOLECULES:
            completed_tasks_results = []  # Store completed 'res' objects.

            # Iterate through the current task list.
            for res, (task_id, start_time) in list(pending_tasks.items()):
                try:
                    # Poll: Use a very short timeout to check if the task is done.
                    result = res.get(timeout=1.0)

                    # --- If code reaches here, the task is complete ---
                    if result:
                        successful_count += 1
                        print(f"✅ [Main] Task {task_id} SUCCEEDED. (Total: {successful_count}/{N_MOLECULES})")
                    else:
                        print(f"❌ [Main] Task {task_id} FAILED (returned False).")

                    completed_tasks_results.append(res)  # Mark for removal.

                except mp.TimeoutError:
                    # --- Task not finished (res.get(0.01) timed out) ---
                    # Check if *our* defined long timeout has been reached.
                    if time.time() - start_time > TASK_TIMEOUT:
                        print(f"⏰ [Main] Task {task_id} TIMED OUT after {TASK_TIMEOUT}s. Worker will be replaced.")
                        completed_tasks_results.append(res)  # Abandon this task, mark for removal.
                    else:
                        # Not yet timed out, just still running, continue waiting.
                        pass

                except Exception as e:
                    # --- Task crashed during execution ---
                    print(f"❌ [Main] Task {task_id} CRASHED: {e}")
                    completed_tasks_results.append(res)  # Mark for removal.

            # --- Clean up completed/timed-out tasks and submit new ones ---
            for res in completed_tasks_results:
                pending_tasks.pop(res)  # Remove from the tracking dictionary.

                # If we still need more molecules, submit a new task.
                if successful_count < N_MOLECULES:
                    submit_task(submitted_count)
                    submitted_count += 1

            # --- Check if we can exit ---
            if successful_count >= N_MOLECULES:
                print(f"Target of {N_MOLECULES} reached. Shutting down pool...")
                pool.terminate()  # Force-terminate all still-running workers.
                break

    print(f"\n✅ Completed: {successful_count}/{N_MOLECULES} molecules succeeded.")


if __name__ == "__main__":
    main()
