import os, random, hashlib, math, itertools, traceback, time
import networkx as nx
import numpy as np
import multiprocessing as mp
import scipy.optimize as opt
from matplotlib import pyplot as plt
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, Draw, rdmolops, rdDepictor, rdchem
from ase import Atoms
from ase.visualize import view

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
    Returns all symmetric equivalent points for an axial coordinate
    based on the specified symmetry.
    """
    q, r = pos
    equivalents = []
    p = (q, r)

    if symmetry == 'C6':
        # Order: 0, -60, -120, -180, -240, -300
        for _ in range(6):
            equivalents.append(p)
            p = (p[0] + p[1], -p[0])
    elif symmetry == 'C3':
        # Order: 0, +120, +240
        for _ in range(3):
            equivalents.append(p)
            p = (-p[0] - p[1], p[0])
    elif symmetry == 'C2':
        # Order: 0, +180
        equivalents.append((q, r))
        equivalents.append((-q, -r))
    elif symmetry == 'mirror':
        # Order: Identity, MirrorOp
        equivalents.append((q, r))
        equivalents.append((-r, -q))
    else:  # C1
        equivalents.append((q, r))

    return equivalents


s60 = math.sqrt(3) / 2.0
c60 = 0.5
c120 = -0.5
s120 = math.sqrt(3) / 2.0

# (q,r) -> (q+r, -q)  (-60 deg)
op_neg60 = lambda v: (c60 * v[0] + s60 * v[1], -s60 * v[0] + c60 * v[1])
# (q,r) -> (-r, q+r)  (+120 deg)
op_120 = lambda v: (c120 * v[0] - s120 * v[1], s120 * v[0] + c120 * v[1])
# (q,r) -> (-q, -r)  (+180 deg)
op_180 = lambda v: (-v[0], -v[1])
# (q,r) -> (-r, -q)  (Mirror across y = -x*sqrt(3))
op_mirror = lambda v: (-c60 * v[0] - s60 * v[1], -s60 * v[0] + c60 * v[1])

op_id = lambda v: (v[0], v[1])  # 0 deg / Identity

TRANSFORM_OPS = {
    'C6': [
        op_id,  # 0
        op_neg60,  # -60
        lambda v: op_neg60(op_neg60(v)),  # -120
        op_180,  # -180
        op_120,  # -240
        lambda v: op_120(op_neg60(v))
    ],
    'C3': [
        op_id,  # 0
        op_120,  # +120
        lambda v: op_120(op_120(v))  # +240
    ],
    'C2': [
        op_id,
        op_180
    ],
    'mirror': [
        op_id,
        op_mirror
    ],
    'asymmetric': [op_id],
    'C1': [op_id]
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


def add_internal_bonds_nx(G, max_dist=1.8):
    """
    NetworkX version of add_internal_bonds. Closes internal rings by
    adding new bonds between nodes with degree 2.
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
            if not H.has_edge(i, j):
                # Calculate distance.
                (x1, y1) = pos[i]
                (x2, y2) = pos[j]
                dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                if dist < max_dist:
                    try:
                        # Calculate the size of the new ring.
                        path_len = nx.shortest_path_length(H, source=i, target=j)
                        new_ring_size = path_len  # path_len is the number of edges.
                        potential_bonds.append({'atoms': (i, j), 'dist': dist, 'ring_size': new_ring_size})
                    except nx.NetworkXNoPath:
                        continue  # Nodes are in different subgraphs.

        if not potential_bonds:
            break

        # Prioritize forming 6-membered rings, then 5-membered, and finally by distance.
        def sort_key(bond_info):
            size = bond_info['ring_size']
            dist = bond_info['dist']
            priority = 2
            if size == 6:
                priority = 0
            elif size == 5:
                priority = 1
            return (priority, dist)

        potential_bonds.sort(key=sort_key)

        for bond_info in potential_bonds:
            i, j = bond_info['atoms']

            # In NetworkX, we just check if the degree will exceed 3.
            if H.degree(i) < 3 and H.degree(j) < 3:
                H.add_edge(i, j)
                bonds_added_in_this_iteration = True
                # After successfully adding one bond, break the inner loop immediately.
                # Because node degrees have changed, degree2_atoms needs to be recalculated.
                break

        if not bonds_added_in_this_iteration:
            break  # No more bonds can be added.

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
                G.add_node(idx, xy=v_cart, elem='C')
                axial_map[idx] = center_ax  # Record which hexagon center this node belongs to.
                node_id_counter += 1
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

def _find_closest_node(G, target_coord, tolerance=1e-3):
    """
    Finds the node ID in G closest to target_coord.
    """
    min_dist_sq = float('inf')
    found_node = None
    tx, ty = target_coord

    # Pre-fetch coordinates for all nodes.
    node_coords = {n: G.nodes[n]['xy'] for n in G.nodes()}

    for node_id, (nx, ny) in node_coords.items():
        dist_sq = (tx - nx) ** 2 + (ty - ny) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            found_node = node_id

    if min_dist_sq < tolerance ** 2:
        return found_node
    else:
        print(f"Warning: No node found for target {target_coord}. Closest was {min_dist_sq ** 0.5} away.")
        return None


def _get_orbital_atoms(G, ref_node, symmetry, axial_map):
    """
    Returns all atom node IDs that are symmetrically equivalent to ref_node
    (based on axial coordinates).
    """
    if ref_node not in axial_map:
        return [ref_node]  # No axial coordinate, fallback to self.

    (q, r) = axial_map[ref_node]  # Atom's center (axial).
    (x, y) = G.nodes[ref_node]['xy']  # Atom's position (Cartesian).
    (cx, cy) = axial_to_cartesian(q, r)  # Atom's center (Cartesian).

    # 1. Calculate the atom's "relative vector" from its center.
    rel_vec = (x - cx, y - cy)

    # 2. Get the *ordered* axial coordinate orbit for the center point.
    centers_axial_orbit = get_symmetric_equivalents_axial((q, r), symmetry)

    # 3. Get the corresponding *ordered* Cartesian vector transformations.
    vector_ops = TRANSFORM_OPS.get(symmetry, [op_id])
    if len(centers_axial_orbit) != len(vector_ops):
        # Safety check: If op list and orbit length mismatch, fallback.
        print(
            f"Warning: Symmetry op mismatch. {symmetry} orbit len {len(centers_axial_orbit)} vs ops len {len(vector_ops)}")
        return [ref_node]

    orbital = []

    # 4. Iterate through symmetry operations.
    for i, center_axial_new in enumerate(centers_axial_orbit):

        # 4a. Get the transformed center (Cartesian).
        (cx_new, cy_new) = axial_to_cartesian(center_axial_new[0], center_axial_new[1])

        # 4b. Get the transformed relative vector (Cartesian).
        op = vector_ops[i]
        (rel_x_new, rel_y_new) = op(rel_vec)

        # 4c. Calculate the target atom's theoretical coordinates.
        target_coord = (cx_new + rel_x_new, cy_new + rel_y_new)

        # 4d. Search for the closest node in the graph.
        node_id = _find_closest_node(G, target_coord)

        if node_id is not None and node_id not in orbital:
            orbital.append(node_id)
        elif node_id is None:
            pass

    return orbital if orbital else [ref_node]


def _hexes(G):
    """Find all 6-membered rings in the graph."""
    try:
        return [c for c in nx.cycle_basis(G) if len(c) == 6]
    except nx.exception.NetworkXNoCycle:
        return []


def stone_wales_bond_switch(G, u, v):
    """Performs a fixed-direction Stone-Wales transformation on edge (u,v). Returns (new_graph, success)."""
    if not G.has_edge(u, v):
        return G, False

    carriers = [cyc for cyc in _hexes(G) if u in cyc and v in cyc]
    if len(carriers) != 2:
        return G, False

    ringA, ringB = carriers

    # Find the four adjacent nodes: ua, va in ringA; ub, vb in ringB.
    def side(a, b, ring):
        ia = ring.index(a)
        ib = ring.index(b)
        if (ia + 1) % 6 == ib:  # a->b order
            return ring[(ia - 1) % 6], ring[(ib + 1) % 6]
        else:  # b->a order
            return ring[(ib - 1) % 6], ring[(ia + 1) % 6]

    ua, va = side(u, v, ringA)
    ub, vb = side(u, v, ringB)

    H = G.copy()
    if not (H.has_edge(u, ua) and H.has_edge(v, vb)):
        return G, False

    # Fixed rotation: break (u,ua)(v,vb), form (u,vb)(v,ua).
    H.remove_edge(u, ua)
    H.remove_edge(v, vb)
    H.add_edge(u, vb)
    H.add_edge(v, ua)

    return H, True


def apply_symmetric_sw(G, symmetry, axial_map, rng=None):
    if rng is None:
        rng = random.Random()

    # 1. Select a seed bond (c1, c2).
    cand = [(u, v) for u, v in G.edges()
            if G.degree(u) == 3 and G.degree(v) == 3]
    if not cand:
        return G
    rng.shuffle(cand)

    for c1, c2 in cand:
        # c3: In the same ring as c1, adjacent to c1, but not c2.
        c3_set = {w for w in G.neighbors(c1) if w != c2 and
                  any(c1 in c and c2 in c and w in c for c in _hexes(G))}
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

    return H


def substitute_symmetric_atoms(G, symmetry, axial_map, elem, num_sites=1, rng=None):
    """
    Pre-generates "substitutable orbits" using the axial_map
    and symmetry operations.
    """
    if rng is None:
        rng = random.Random()

    degree_map = {'O': 2, 'N': 3, 'B': 3}
    target_deg = degree_map.get(elem)
    if target_deg is None:
        return G

    H = G.copy()

    # 1. Pre-generate all "substitutable" orbits.
    from collections import defaultdict
    orbit_dict = defaultdict(list)  # key: (axial_center, degree, elem) -> list of orbits(frozenset of nodeIDs)

    for node in H.nodes():
        if H.nodes[node]['elem'] != 'C' or H.degree(node) != target_deg:
            continue
        if node not in axial_map:  # Some dangling nodes might not have a mapping.
            continue
        axial_center = axial_map[node]
        # Get the full orbit.
        orb = _get_orbital_atoms(H, node, symmetry, axial_map)
        if not orb:
            continue
        # Use frozenset for deduplication & ordering, ensuring the same hash for the same orbit.
        orb_set = frozenset(orb)
        key = (axial_center, target_deg, 'C')
        # Store only once to avoid duplicates.
        if orb_set not in orbit_dict[key]:
            orbit_dict[key].append(orb_set)

    # 2. Flatten into one large list for easy random selection.
    all_orbits = []
    for orbits in orbit_dict.values():
        all_orbits.extend(orbits)
    if not all_orbits:
        return H

    rng.shuffle(all_orbits)

    # 3. Select mutually disjoint orbits.
    replaced_nodes = set()
    used = 0
    for orb_set in all_orbits:
        if used >= num_sites:
            break
        # Must be disjoint from already replaced nodes.
        if orb_set.isdisjoint(replaced_nodes):
            for n in orb_set:
                H.nodes[n]['elem'] = elem
            replaced_nodes.update(orb_set)
            used += 1

    return H


# =============================================================================
# PART 3: GRAPH TO MOLECULE CONVERSION
# Core function: Convert a NetworkX graph (with heteroatoms) to an RDKit
# molecule and perform post-processing.
# =============================================================================

def optimize_2d_geometry(mol, k_bond=20000.0, k_angle=15000.0,
                         bond_range=(1.38, 1.45), target_angle_deg=120.0,
                         max_steps=6000):
    """
    Optimizes 2D coordinates using soft constraints with tolerance.
    Encourages 120° bond angles.
    """
    from scipy.optimize import minimize
    import numpy as np
    import math

    num_atoms = mol.GetNumAtoms()
    if num_atoms < 2:
        return mol

    try:
        conf = mol.GetConformer()
    except ValueError:
        return mol

    # view 3d conformer
    positions = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atoms = Atoms(numbers=atomic_numbers, positions=positions)
    view(atoms)


    # Initial 2D coordinates.
    pos = np.array([conf.GetAtomPosition(i) for i in range(num_atoms)])[:, :2]
    x0 = pos.flatten()

    # Extract bonds and angles.
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]

    angles = []
    for atom in mol.GetAtoms():
        j = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(neighbors) >= 2:
            for i, k in itertools.combinations(neighbors, 2):
                angles.append((i, j, k))

    theta0_rad = math.radians(target_angle_deg)
    l_min, l_max = bond_range

    def energy_fn(flat_coords):
        coords = flat_coords.reshape((num_atoms, 2))
        E = 0.0

        # Bond length energy (soft range penalty).
        for i, j in bonds:
            dx = coords[i] - coords[j]
            length = np.hypot(dx[0], dx[1])
            if length < 1e-6:
                E += 1e4  # Avoid overlap.
                continue
            if length < l_min:
                E += 0.5 * k_bond * (length - l_min) ** 2
            elif length > l_max:
                E += 0.5 * k_bond * (length - l_max) ** 2

        # Bond angle energy (encourage 120°).
        for i, j, k in angles:
            v_ji = coords[i] - coords[j]
            v_jk = coords[k] - coords[j]
            norm_ji = np.hypot(v_ji[0], v_ji[1])
            norm_jk = np.hypot(v_jk[0], v_jk[1])
            if norm_ji < 1e-6 or norm_jk < 1e-6:
                E += 0.5 * k_angle * (math.pi - theta0_rad) ** 2
                continue
            cos_theta = np.clip(np.dot(v_ji, v_jk) / (norm_ji * norm_jk), -1.0, 1.0)
            theta = math.acos(cos_theta)
            E += 0.5 * k_angle * (theta - theta0_rad) ** 2

        return E

    # # Run optimization.
    # result = minimize(energy_fn, x0, method='L-BFGS-B',
    #                   options={'maxiter': max_steps, 'disp': False})
    #
    # if not result.success:
    #     print(f"[WARN] 2D optimization did not converge: {result.message}")
    #
    # # Write coordinates back.
    # optimized = result.x.reshape((num_atoms, 2))
    # for i in range(num_atoms):
    #     conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(optimized[i][0], optimized[i][1], 0.0))
    # conf.Set3D(False)
    # rdDepictor.StraightenDepiction(mol)
    return mol


def graph_with_hetero_to_mol(G):
    """
    Creates an RDKit molecule from a NetworkX graph,
    including index correction and chemical sanity checks to prevent library crashes.
    """
    mol = Chem.RWMol()
    valence = {'C': 4, 'N': 3, 'O': 2, 'B': 3}
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
        Chem.SanitizeMol(mol)
    except Exception as e:
        # If the check fails, the molecule is problematic. Mark as invalid and skip.
        print(f"[WARN] RDKit sanitization failed for a molecule: {e}. Skipping.")
        return None

    # # 4. Add hydrogens and post-process.
    # try:
    #     mol = mol.GetMol()
    #     if mol is None: return None
    #
    #     for atom in mol.GetAtoms():
    #         deg = atom.GetDegree()
    #         wanted = valence.get(atom.GetSymbol(), 4)
    #         atom.SetNumExplicitHs(max(0, wanted - deg))
    #         atom.UpdatePropertyCache()
    #
    #     rdDepictor.Compute2DCoords(mol)
    #     rdDepictor.StraightenDepiction(mol)
    #     mol = optimize_2d_geometry(mol, bond_range=(1.38, 1.45))
    # except Exception as e:
    #     print(f"[WARN] Post-sanitization processing failed: {e}")
    #     return None  # If subsequent steps also fail, return None.

    return mol


def get_structure_from_mol2d(mol_2d):
    """
    2D -> 3D: On failure, outputs the 2D optimized structure (Z=0).
    Returns an ASE Atoms object (either 3D or 2D+Z=0).
    """
    ...

    mol = assign_kekule_by_matching(mol_2d)
    mol = charge_carbons_without_double(mol)

    smi = Chem.MolToSmiles(mol, canonical=True)
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)

    # Add hydrogens
    mol = Chem.AddHs(mol)

    show_mol(mol)

    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, params=AllChem.ETKDGv3())

    # Extract atomic numbers
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    # Get a random conformer
    conformer = mol.GetConformer(0)

    # Extract coordinates
    coords = [conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    coords = np.array([[pos.x, pos.y, pos.z] for pos in coords])

    # Create an ASE Atoms object
    ase_atoms = Atoms(numbers=atomic_numbers, positions=coords)
    # RDKit canonical SMILES (aromatic)

    smi = Chem.MolToSmiles(mol, canonical=True)
    view(ase_atoms)
    return ase_atoms


# =============================================================================
# PART 4: UTILITIES & EXPORT
# Core function: Helper functions for display, saving, hash calculation, etc.
# =============================================================================

def get_canonical_smiles_hash(mol):
    """Returns the SHA256 of the Canonical SMILES (used for deduplication/naming)."""
    smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    return hashlib.sha256(smi.encode()).hexdigest()[:8]


# 1. Check for atoms that are too close.
def has_too_close_atoms(atoms, dmin=0.8):
    d = atoms.get_all_distances(mic=False)
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] < dmin:
                return True
    return False


# 2. Check for crossing bonds (2D coordinates are sufficient).
def has_crossing_bonds(G):
    """
    Simple scan of all bond pairs to see if line segments intersect (excluding common endpoints).
    Only for 2D graphs, fast.
    """
    from itertools import combinations
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])

    def seg_intersect(p1, p2, p3, p4):
        # Does p1-p2 intersect with p3-p4 (excluding endpoints)?
        c1, c2 = ccw(p1, p2, p3), ccw(p1, p2, p4)
        c3, c4 = ccw(p3, p4, p1), ccw(p3, p4, p2)
        return (c1 * c2 < 0) and (c3 * c4 < 0)

    pos = {n: G.nodes[n]['xy'] for n in G.nodes()}
    edges = list(G.edges())
    for (u1, v1), (u2, v2) in combinations(edges, 2):
        if len({u1, v1, u2, v2}) < 4:  # Has common endpoint, skip.
            continue
        p1, p2 = pos[u1], pos[v1]
        p3, p4 = pos[u2], pos[v2]
        if seg_intersect(p1, p2, p3, p4):
            return True
    return False


# 3. Check for nested rings (using NetworkX cycle_basis).
def has_nested_rings(G, dmin=1.3):
    from networkx import cycle_basis
    pos = {n: G.nodes[n]['xy'] for n in G.nodes()}  # 2D coordinates
    cycles = [c for c in cycle_basis(G) if 4 <= len(c) <= 8]
    if len(cycles) < 2:
        return False

    # Calculate the center of each ring.
    centers = []
    for c in cycles:
        pts = np.array([pos[n] for n in c])
        centers.append(pts.mean(axis=0))  # shape (2,)

    # Check distances pair-wise.
    for i, c1 in enumerate(centers):
        for j, c2 in enumerate(centers):
            if i >= j:
                continue
            if np.hypot(c1[0] - c2[0], c1[1] - c2[1]) < dmin:
                return True
    return False


def save_graph_svg(G, path, size=(400, 400)):
    """Saves a NetworkX graph as an SVG."""
    pos = {n: G.nodes[n]['xy'] for n in G.nodes()}
    plt.figure(figsize=(size[0] / 100, size[1] / 100))
    nx.draw(G, pos, node_size=50, node_color='black', width=1.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

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

def run_one(task_id, total_target, seed, N_SITES, O_SITES, B_SITES, do_N=True, do_O=True, do_B=True):
    """Wraps the single molecule generation logic to be callable by multiprocessing."""
    try:
        rng = random.Random(seed)
        # [MOD] Update print statements to reflect task ID
        print(f"\n[Worker {os.getpid()}] --- Starting task {task_id} (Target: {total_target} successes) ---")
        symmetries = ['C2', 'C3', 'C6', 'mirror', 'asymmetric']
        target_symmetry = rng.choice(symmetries)
        num_rings = rng.randint(10, 80)
        print(f"Target symmetry: {target_symmetry}, rings: ~{num_rings}")

        centers = grow_random_symmetric(steps=num_rings, symmetry=target_symmetry, rng=rng)
        if not centers:
            print(f"[Task {task_id}] Failed to generate centers. Skipping.")
            return False

        flake_graph, axial_map = centers_to_graph(centers)

        # Call the NetworkX version of internal bond addition here.
        flake_graph = add_internal_bonds_nx(flake_graph)

        show_graph(flake_graph)

        initial_symmetry = detect_highest_symmetry(flake_graph, axial_map)

        # try:
        #     flake_mol_2d = graph_with_hetero_to_mol(flake_graph)
        #     if flake_mol_2d is None or flake_mol_2d.GetNumAtoms() == 0:
        #         print(f"[Task {task_id}] Failed to generate flake RDKit molecule. Skipping.")
        #         return False
        #
        # except Exception as e:
        #     print(f"[Task {task_id}] Error generating flake molecule: {e}")
        #     return False

        # Calculate the hash after closing the rings.
        # hash_prefix = get_canonical_smiles_hash(flake_mol_2d)
        # print(f"[Task {task_id}] Initial actual symmetry: {initial_symmetry}. Hash: {hash_prefix}")

        # out_path is the directory we will always use.
        out_path = os.path.join("symmetric_molecules", initial_symmetry)
        os.makedirs(out_path, exist_ok=True)

        defective_graph = apply_symmetric_sw(flake_graph, initial_symmetry, axial_map, rng=rng)
        final_graph = defective_graph
        if do_N:
            final_graph = substitute_symmetric_atoms(final_graph, initial_symmetry, axial_map, elem='N', num_sites=N_SITES,
                                                     rng=rng)
        if do_O:
            final_graph = substitute_symmetric_atoms(final_graph, initial_symmetry, axial_map, elem='O', num_sites=O_SITES,
                                                     rng=rng)
        if do_B:
            final_graph = substitute_symmetric_atoms(final_graph, initial_symmetry, axial_map, elem='B', num_sites=B_SITES,
                                                     rng=rng)

        show_graph(final_graph)

        # final_symmetry_actual = detect_highest_symmetry(final_graph, axial_map)

        # print(f"[Task {task_id}] Final actual symmetry: {final_symmetry_actual}")

        final_mol_2d = graph_with_hetero_to_mol(final_graph)
        if final_mol_2d is None or final_mol_2d.GetNumAtoms() == 0:
            print(f"[Task {task_id}] Failed to generate final RDKit molecule. Skipping.")
            return False

        show_mol(final_mol_2d)

        final_ase_atoms = get_structure_from_mol2d(final_mol_2d)
        if final_ase_atoms is None:
            print(f"[Task {task_id}] Skipping this molecule due to 3D embedding failure.")
            return False
        if has_too_close_atoms(final_ase_atoms):
            print(f"[Task {task_id}] Rejected: too close atoms.")
            return False
        if has_crossing_bonds(final_graph):
            print(f"[Task {task_id}] Rejected: crossing bonds detected.")
            return False
        if has_nested_rings(final_graph):
            print(f"[Task {task_id}] Rejected: nested rings detected.")
            return False

        # ---------- Count 4~5~6~7~8 membered rings ----------
        ring_info = final_mol_2d.GetRingInfo()
        atom_rings = ring_info.AtomRings()

        ring_count = {k: 0 for k in range(4, 9)}
        for ring in atom_rings:
            k = len(ring)
            if 4 <= k <= 8:
                ring_count[k] += 1

        name_tag = "_".join(f"{ring_count[k]}" for k in range(4, 9))  # n4_n5_n6_n7_n8
        flake_hash16 = get_canonical_smiles_hash(flake_mol_2d)[:16]  # First 16 chars of the flake's hash.
        base_name = f"{name_tag}_{flake_hash16}"

        # ---------- Save both flake & final ----------
        out_path = os.path.join("symmetric_molecules", initial_symmetry)
        os.makedirs(out_path, exist_ok=True)

        # flake
        save_graph_svg(flake_graph, os.path.join(out_path, f"{base_name}_flake.svg"))
        save_graph_xyz(flake_graph, os.path.join(out_path, f"{base_name}_flake.xyz"))
        save_mol_file(flake_mol_2d, os.path.join(out_path, f"{base_name}_flake.mol"))
        # final
        save_mol_svg(final_mol_2d, os.path.join(out_path, f"{base_name}_final.svg"))
        save_ase_xyz(final_ase_atoms, os.path.join(out_path, f"{base_name}_final.xyz"),
                     comment=f"Task {task_id} final")
        save_mol_file(final_mol_2d, os.path.join(out_path, f"{base_name}_final.mol"))
        final_smi = Chem.MolToSmiles(final_mol_2d, isomericSmiles=False, canonical=True)
        print(f"✅ [Task {task_id}] Successfully saved {base_name} SMILES: {final_smi}")
        return True

    except Exception as e:
        print(f"[Worker {os.getpid()}] Unhandled exception in task {task_id}: {e}")
        traceback.print_exc()
        return False


def main():
    N_MOLECULES = 1  # This is the target number of successes.
    N_SWITCH = 1
    O_SWITCH = 0
    B_SWITCH = 1
    N_SITES = 2
    O_SITES = 0
    B_SITES = 2
    TASK_TIMEOUT = 600  # Timeout for a single task (seconds).
    pool_size = 1
    print(f"Using pool size: {pool_size}. Target: {N_MOLECULES} successes. Timeout: {TASK_TIMEOUT}s")

    rng_master = random.Random()
    # Dictionary to track running tasks: {AsyncResult: (task_id, start_time)}
    pending_tasks = {}
    successful_count = 0
    submitted_count = 0

    # maxtasksperchild=1 ensures a worker is replaced after one task, preventing memory leaks or hangs.
    with mp.Pool(processes=pool_size, maxtasksperchild=1) as pool:

        # Helper function to submit a new task.
        def submit_task(idx):
            seed = rng_master.randrange(1, 2 ** 20)
            args = (idx, N_MOLECULES, seed)
            res = pool.apply_async(run_one, args=(idx, N_MOLECULES, seed,N_SITES, O_SITES, B_SITES,
                                      bool(N_SWITCH), bool(O_SWITCH), bool(B_SWITCH) ))
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
                    result = res.get(timeout=0.01)

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

            # --- Prevent busy-waiting ---
            if not completed_tasks_results:
                time.sleep(1)  # If no tasks completed this round, sleep for 1s before checking again.

    print(f"\n✅ Completed: {successful_count}/{N_MOLECULES} molecules succeeded.")


def show_mol(mol):
    img = Draw.MolToImage(mol, size=(300,300))
    img.show()


def assign_kekule_by_matching(mol):
    """
    Set an approximate Kekulé pattern:
    - consider only C(sp2)-C(sp2) bonds (ignore N/O and any hetero bonds)
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
    # Chem.SanitizeMol(mol2)  # valence/conjugation checks

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

def show_graph(graph):
    pos = {n: graph.nodes[n]['xy'] for n in graph.nodes()}
    nx.draw(graph, pos, with_labels=True, node_size=100)
    plt.show()

if __name__ == "__main__":
    main()