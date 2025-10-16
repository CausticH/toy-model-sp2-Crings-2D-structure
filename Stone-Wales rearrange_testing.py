import os, random, hashlib, math, itertools, traceback
from os import cpu_count
import networkx as nx
import numpy as np
import multiprocessing as mp
from matplotlib import pyplot as plt
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, Draw, rdmolops, rdDepictor, rdchem
from ase import Atoms
from ase.visualize import view

# =============================================================================
# PART 1: SYMMETRIC FLAKE GENERATION (from C6-symmetric_growth_v1.py)
# 核心功能：基于轴坐标系生成具有特定点群对称性的六边形中心点集合。
# =============================================================================

def axial_to_cartesian(q, r, size=1.42):
    """将六边形的轴坐标 (q, r) 转换为笛卡尔坐标。"""
    x = size * math.sqrt(3) * (q + r / 2.0)
    y = size * 3.0 / 2.0 * r
    return (x, y)


def hex_corner(center, size, i):
    """返回六边形第 i 个角的笛卡尔坐标。"""
    (cx, cy) = center
    angle_deg = 60 * i + 30
    angle_rad = math.radians(angle_deg)
    return (cx + size * math.cos(angle_rad), cy + size * math.sin(angle_rad))


def get_symmetric_equivalents_axial(pos, symmetry):
    """根据对称性，返回一个轴坐标的所有对称等效点。"""
    q, r = pos
    # C6 旋转: (q, r) -> (q+r, -q)
    # C3 旋转: (q, r) -> (-r, q+r) -> (q+r, -q) ...
    # C2 旋转: (q, r) -> (-q, -r)
    # 镜面 (沿 q+r=0): (q, r) -> (-r, -q)
    equivalents = set()
    p = (q, r)
    if symmetry == 'C6':
        for _ in range(6):
            equivalents.add(p)
            p = (p[0] + p[1], -p[0])
    elif symmetry == 'C3':
        for _ in range(3):
            equivalents.add(p)
            p = (-p[1], p[0] + p[1])
    elif symmetry == 'C2':
        equivalents.add((q, r))
        equivalents.add((-q, -r))
    elif symmetry == 'mirror':
        equivalents.add((q, r))
        equivalents.add((-r, -q))
    else:  # C1
        equivalents.add((q, r))
    return list(equivalents)


def grow_random_symmetric(steps, symmetry, seed_centers=None, rng=None):
    """
    按指定的对称性进行随机生长
    """
    if rng is None:
        rng = random.Random()
    if seed_centers is None:
        seed_centers = [(0, 0)]  # 对称性生长必须从中心开始

    centers = set()
    # 初始点也必须符合对称性
    initial_equivalents = get_symmetric_equivalents_axial(seed_centers[0], symmetry)
    for p in initial_equivalents:
        centers.add(p)

    nbr_dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    # 循环直到达到目标环数
    while len(centers) < steps + 1:
        # 1. 找出所有可能的生长点
        candidates = set()
        for c in centers:
            for d in nbr_dirs:
                pos = (c[0] + d[0], c[1] + d[1])
                if pos not in centers:
                    candidates.add(pos)

        if not candidates:
            break

        # 2. 随机化候选点顺序进行尝试
        candidate_list = list(candidates)
        rng.shuffle(candidate_list)

        added_successfully = False
        for new in candidate_list:
            # 获取当前候选点的所有对称等效点
            new_points = get_symmetric_equivalents_axial(new, symmetry)
            new_points_set = set(new_points)

            # --- 检查所有新点是否都与现有分子邻接 ---
            all_connected = True
            for p in new_points_set:
                # 检查点 p 是否至少有一个邻居在 centers 集合中
                is_p_connected = False
                for d in nbr_dirs:
                    neighbor = (p[0] + d[0], p[1] + d[1])
                    # 检查邻居是否在现有结构中 或 在即将添加的新点中
                    if neighbor in centers:
                        is_p_connected = True
                        break

                # 如果这个点找不到任何与现有结构的连接，则这组点无效
                if not is_p_connected:
                    all_connected = False
                    break

            # 3. 如果所有点都验证通过，则添加它们并进入下一轮生长
            if all_connected:
                for p in new_points_set:
                    centers.add(p)
                added_successfully = True
                break  # 成功添加，跳出对候选点的遍历

        # 如果遍历了所有候选点都找不到能保证连通性的生长方式，则停止
        if not added_successfully:
            print("Warning: Could not find a valid connected growth point. Stopping.")
            break

    return sorted(list(centers))


def add_internal_bonds(mol, max_dist=1.8):
    """
    通过在度为2的原子间添加新键来封闭内部的环。
    """
    if mol.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(mol)

    while True:
        bonds_added_in_this_iteration = False
        rw_mol = Chem.RWMol(mol)
        conf = rw_mol.GetConformer()

        degree2_atoms_indices = [a.GetIdx() for a in rw_mol.GetAtoms() if a.GetDegree() == 2]
        if len(degree2_atoms_indices) < 2: break

        potential_bonds = []
        for i, j in itertools.combinations(degree2_atoms_indices, 2):
            if rw_mol.GetBondBetweenAtoms(i, j) is None:
                dist = conf.GetAtomPosition(i).Distance(conf.GetAtomPosition(j))
                if dist < max_dist:
                    path = rdmolops.GetShortestPath(rw_mol, i, j)
                    if not path: continue
                    new_ring_size = len(path)
                    potential_bonds.append({'atoms': (i, j), 'dist': dist, 'ring_size': new_ring_size})

        if not potential_bonds: break

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
            temp_mol = Chem.RWMol(rw_mol)
            temp_mol.AddBond(i, j, Chem.BondType.SINGLE)

            # --- 预检查（改回使用旧版函数以保证兼容性） ---
            high_valence_found = False
            atom_i = temp_mol.GetAtomWithIdx(i)
            atom_j = temp_mol.GetAtomWithIdx(j)

            if atom_i.GetValence(Chem.ValenceType.EXPLICIT) > 4 or atom_j.GetValence(Chem.ValenceType.EXPLICIT) > 4:
                high_valence_found = True

            if high_valence_found:
                continue
            # --- 预检查结束 ---

            with rdBase.BlockLogs():
                try:
                    Chem.SanitizeMol(temp_mol)
                    mol = temp_mol.GetMol()
                    bonds_added_in_this_iteration = True
                    break
                except Exception:
                    continue

        if not bonds_added_in_this_iteration:
            break

    rdDepictor.Compute2DCoords(mol)
    return mol


def detect_highest_symmetry(G, axial_map):
    """基于节点坐标+元素，检测图的实际最高对称性"""
    nodes = list(G.nodes())
    if not nodes:
        return 'asymmetric'

    # 构建 (轴坐标, 元素) 集合
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
    """返回 (G, axial_map) """
    G = nx.Graph()
    vertex_map = {}
    node_id_counter = 0
    axial_map = {}  # 节点ID → 轴坐标 (q,r)

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
                axial_map[idx] = center_ax  # 记录这个节点属于哪个六边形中心
                node_id_counter += 1
            vert_indices.append(vertex_map[rv_cart])

        for i in range(6):
            u, v = vert_indices[i], vert_indices[(i + 1) % 6]
            if not G.has_edge(u, v):
                G.add_edge(u, v)

    # 移除悬挂键
    dangling = [n for n, d in G.degree() if d <= 1]
    while dangling:
        G.remove_nodes_from(dangling)
        for n in dangling:
            axial_map.pop(n, None)
        dangling = [n for n, d in G.degree() if d <= 1]

    return G, axial_map


# =============================================================================
# PART 2: DEFECT APPLICATION (from mod_hex.py and atom_substitude.py)
# 核心功能：在NetworkX图上施加Stone-Wales变换和原子替换。
# 已修改为对称性操作。
# =============================================================================

def _get_orbital_atoms(G, ref_node, symmetry, axial_map):
    """返回与 ref_node 对称等效的所有原子节点ID（基于轴坐标）"""
    if ref_node not in axial_map:
        return [ref_node]  # 无轴坐标，退化为自身

    q, r = axial_map[ref_node]
    equivalents_axial = get_symmetric_equivalents_axial((q, r), symmetry)
    orbital = []

    # 预建笛卡尔坐标 → 节点ID 映射
    cart_to_node = {G.nodes[n]['xy']: n for n in G.nodes()}

    for eq_q, eq_r in equivalents_axial:
        x, y = axial_to_cartesian(eq_q, eq_r)
        x_rounded = round(x, 6)
        y_rounded = round(y, 6)
        if (x_rounded, y_rounded) in cart_to_node:
            orbital.append(cart_to_node[(x_rounded, y_rounded)])

    return orbital if orbital else [ref_node]


def _hexes(G):
    """找到图中所有的六元环。"""
    try:
        return [c for c in nx.cycle_basis(G) if len(c) == 6]
    except nx.exception.NetworkXNoCycle:
        return []


def stone_wales_bond_switch(G, u, v):
    """对(u,v)键执行Stone-Wales变换。"""
    if not G.has_edge(u, v): return G, False

    # 找到共享(u,v)的两个六元环
    carriers = [cyc for cyc in _hexes(G) if u in cyc and v in cyc and G.has_edge(u, v)]
    if len(carriers) != 2: return G, False

    H = G.copy()
    # 找到u和v在两个环上的邻居
    neighbors_u = list(H.neighbors(u))
    neighbors_v = list(H.neighbors(v))

    u_side1, v_side1 = None, None
    for i in range(6):
        if carriers[0][i] == u and carriers[0][(i + 1) % 6] == v:
            u_side1, v_side1 = carriers[0][(i - 1) % 6], carriers[0][(i + 2) % 6]
            break
        if carriers[0][i] == v and carriers[0][(i + 1) % 6] == u:
            v_side1, u_side1 = carriers[0][(i - 1) % 6], carriers[0][(i + 2) % 6]
            break

    u_side2, v_side2 = None, None
    for i in range(6):
        if carriers[1][i] == u and carriers[1][(i + 1) % 6] == v:
            u_side2, v_side2 = carriers[1][(i - 1) % 6], carriers[1][(i + 2) % 6]
            break
        if carriers[1][i] == v and carriers[1][(i + 1) % 6] == u:
            v_side2, u_side2 = carriers[1][(i - 1) % 6], carriers[1][(i + 2) % 6]
            break

    if any(x is None for x in [u_side1, v_side1, u_side2, v_side2]): return G, False

    # 旋转键：断开 u-v_side1 和 v-u_side2，连接 u-u_side2 和 v-v_side1 (这是一种旋转方式)
    if H.has_edge(u, v_side1) and H.has_edge(v, u_side2):
        H.remove_edge(u, v_side1)
        H.remove_edge(v, u_side2)
        H.add_edge(u, u_side2)
        H.add_edge(v, v_side1)
        return H, True

    return G, False


def apply_symmetric_sw(G, symmetry, axial_map, rng=None):
    """严格对称版 SW：按 C1-C2-C3-C4 规则选边，再旋转所有等效边"""
    if rng is None:
        rng = random.Random()

    # ---- 2.1 选一条“种子边”(C1,C2) ----
    cand_edges = [(u, v) for u, v in G.edges()
                  if G.degree(u) == 3 and G.degree(v) == 3]
    if not cand_edges: return G
    rng.shuffle(cand_edges)

    for c1, c2 in cand_edges:
        # C3：与 C1 同环且相邻
        c3_cand = {w for w in G.neighbors(c1) if w != c2 and
                   any(c1 in c and c2 in c and w in c for c in _hexes(G))}
        if not c3_cand:
            continue
        c3 = rng.choice(list(c3_cand))

        # C4：与 C2 相邻但不在 C1-C2-C3 所在六元环
        avoid_rings = [c for c in _hexes(G) if c1 in c and c2 in c]
        c4_cand = {w for w in G.neighbors(c2) if w != c1 and
                   not any(w in c for c in avoid_rings)}
        if not c4_cand:
            continue
        c4 = rng.choice(list(c4_cand))
        break
    else:
        return G  # 没找到合适边

    # ---- 2.2 找出这条边的“轨道” ----
    # 我们让 (C1,C2) 及其对称等效边同时旋转
    # 先找 C1 的轨道
    c1_orb = _get_orbital_atoms(G, c1, symmetry,axial_map)
    c2_orb = _get_orbital_atoms(G, c2, symmetry,axial_map)
    c3_orb = _get_orbital_atoms(G, c3, symmetry,axial_map)
    c4_orb = _get_orbital_atoms(G, c4, symmetry,axial_map)

    # 建立 (c1,c2,c3,c4) → 所有等效四元组
    # 简单做法：按顺序一一对应（轨道长度必须相同，否则跳过）
    if not (len(c1_orb) == len(c2_orb) == len(c3_orb) == len(c4_orb)):
        return G

    # ---- 2.3 批量旋转 ----
    H = G.copy()
    for c1x, c2x, c3x, c4x in zip(c1_orb, c2_orb, c3_orb, c4_orb):
        # 必须仍满足边存在 & 度=3
        if not (H.has_edge(c1x, c2x) and H.degree(c1x) == H.degree(c2x) == 3):
            continue
        # 断开 c1x-c3x  & c2x-c4x
        if not (H.has_edge(c1x, c3x) and H.has_edge(c2x, c4x)):
            continue
        H.remove_edge(c1x, c3x)
        H.remove_edge(c2x, c4x)
        # 新建 c1x-c4x  & c2x-c3x
        H.add_edge(c1x, c4x)
        H.add_edge(c2x, c3x)

    if nx.is_connected(H):
        return H
    return G


def substitute_symmetric_atoms(G, symmetry, axial_map, elem, num_sites=1, rng=None):
    """
    对称替换杂原子，O 只接 deg=2 的碳，N/B 只接 deg=3 的碳。
    返回新图（副本）。
    """
    if rng is None:
        rng = random.Random()

    degree_map = {'O': 2, 'N': 3, 'B': 3}
    target_deg = degree_map.get(elem)
    if target_deg is None:
        return G

    H = G.copy()
    # 按元素选候选：O 要 deg=2，N/B 要 deg=3
    candidates = [n for n in H.nodes()
                  if H.degree(n) == target_deg and H.nodes[n]['elem'] == 'C']
    if not candidates:
        return H

    rng.shuffle(candidates)
    replaced_orbit = set()
    for _ in range(num_sites):
        for can in candidates:
            if can in replaced_orbit:
                continue
            orbit = _get_orbital_atoms(H, can, symmetry, axial_map)
            for n in orbit:
                H.nodes[n]['elem'] = elem
            replaced_orbit.update(orbit)
            break
    return H

# =============================================================================
# PART 3: GRAPH TO MOLECULE CONVERSION (from test.py)
# 核心功能：将带有杂原子的NetworkX图转换为RDKit分子，并进行后处理。
# =============================================================================

def graph_with_hetero_to_mol(G):
    """
    从NetworkX图创建RDKit分子，
    包含索引修正和化学合理性检查，防止底层库崩溃。
    """
    mol = Chem.RWMol()
    valence = {'C': 4, 'N': 3, 'O': 2, 'B': 3}
    node_to_idx = {}

    # 1. 添加原子并填充映射
    for n in G.nodes():
        elem = G.nodes[n].get('elem', 'C')
        if not isinstance(elem, str) or not elem.strip(): elem = 'C'
        elem = elem.strip()[0].upper()
        if elem not in valence: elem = 'C'
        atom = Chem.Atom(elem)
        atom.SetNoImplicit(True)
        idx = mol.AddAtom(atom)
        node_to_idx[n] = idx

    # 2. 使用映射添加正确的化学键
    for u, v in G.edges():
        idx_u = node_to_idx.get(u)
        idx_v = node_to_idx.get(v)
        if idx_u is not None and idx_v is not None:
            mol.AddBond(idx_u, idx_v, Chem.BondType.SINGLE)

    # 3.在任何操作前进行化学合理性检查 ***
    try:
        # SanitizeMol会检查价态等化学规则。如果图结构不合法，这里会抛出异常。
        Chem.SanitizeMol(mol)
    except Exception as e:
        # 如果检查失败，说明这个分子结构有问题，我们将其标记为无效并跳过。
        # 这样就将底层的崩溃转换为了可控的Python异常。
        # print(f"[WARN] RDKit sanitization failed for a molecule: {e}. Skipping.")
        return None

    # 4. 补氢和后续处理
    try:
        mol = mol.GetMol()
        if mol is None: return None

        for atom in mol.GetAtoms():
            deg = atom.GetDegree()
            wanted = valence.get(atom.GetSymbol(), 4)
            atom.SetNumExplicitHs(max(0, wanted - deg))
            atom.UpdatePropertyCache()

        rdDepictor.Compute2DCoords(mol)
        rdDepictor.StraightenDepiction(mol)
    except Exception as e:
        print(f"[WARN] Post-sanitization processing failed: {e}")
        return None  # 如果后续步骤依然出错，也返回None

    return mol


def get_structure_from_mol2d(mol_2d):
    """
    2D → 3D：失败则输出 2D 优化结构（Z=0）
    返回 ASE Atoms 对象（3D 或 2D+Z=0）
    """
    mol = Chem.AddHs(mol_2d)

    # ---------- 1. 获取 2D 坐标 ----------
    conf2d = mol.GetConformer()
    pos2d = np.array([conf2d.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])

    # ---------- 2. 尝试 3D 嵌入 ----------
    params = AllChem.ETKDGv3()
    params.useRandomCoords = False  # 用 2D+Z 作为初始
    params.maxIterations = 500
    params.randomSeed = 42

    if AllChem.EmbedMolecule(mol, params) == 0:
        try:
            AllChem.UFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
            atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            coords = mol.GetConformer(0).GetPositions()
            return Atoms(numbers=atomic_numbers, positions=coords)
        except:
            pass  # UFF 失败也降级

    # ---------- 3. 降级：输出 2D 结构（Z=0） ----------
    mol = Chem.RemoveHs(mol)
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    coords_2d = np.array([
        [conf2d.GetAtomPosition(i).x, conf2d.GetAtomPosition(i).y, 0.0]
        for i in range(mol.GetNumAtoms())
    ])
    return Atoms(numbers=atomic_numbers, positions=coords_2d)


# =============================================================================
# PART 4: UTILITIES & EXPORT
# 核心功能：显示、保存、计算哈希等辅助函数。
# =============================================================================

def get_canonical_smiles_hash(mol):
    """返回Canonical SMILES的SHA256（用于去重命名）"""
    smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    return hashlib.sha256(smi.encode()).hexdigest()[:8]

def save_graph_svg(G, path, size=(400, 400)):
    """将NetworkX图保存为SVG。"""
    pos = {n: G.nodes[n]['xy'] for n in G.nodes()}
    plt.figure(figsize=(size[0] / 100, size[1] / 100))
    nx.draw(G, pos, node_size=50, node_color='black', width=1.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def save_graph_xyz(G, path, comment=""):
    """将带2D坐标的NetworkX图保存为XYZ文件。"""
    with open(path, 'w') as f:
        f.write(f"{G.number_of_nodes()}\n")
        f.write(f"{comment}\n")
        for n in sorted(G.nodes()):
            elem = G.nodes[n].get('elem', 'C')
            x, y = G.nodes[n]['xy']
            f.write(f"{elem} {x: .6f} {y: .6f} {0.0: .6f}\n")


def save_mol_svg(mol, path, size=(400, 400)):
    """将RDKit分子保存为SVG。"""
    # 为分子生成2D坐标
    mol_copy = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(mol_copy)
    Draw.MolToFile(mol_copy, path, size=size)


def save_ase_xyz(atoms, path, comment=""):
    """将ASE Atoms对象保存为XYZ文件。"""
    with open(path, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{comment}\n")
        for sym, pos in zip(atoms.get_chemical_symbols(), atoms.positions):
            f.write(f"{sym} {pos[0]: .6f} {pos[1]: .6f} {pos[2]: .6f}\n")


def is_geometry_reasonable(atoms_3d, planarity_threshold=0.2, min_bond=1.2, max_bond=1.7):
    """
    检查优化后的3D ASE Atoms对象的几何结构是否合理。
    如果几何结构合理则返回 True，否则返回 False。
    """
    if len(atoms_3d) < 2:
        return True

    # 1. 检查平面性
    positions = atoms_3d.get_positions()
    z_coords = positions[:, 2]
    if np.std(z_coords) > planarity_threshold:
        # print(f"[Reject] Structure is not planar, Z-stddev: {np.std(z_coords):.2f} Å")
        return False

    # 2. 检查键长
    dists = atoms_3d.get_all_distances(mic=True)
    # 遍历上三角矩阵以避免重复计算
    for i in range(len(atoms_3d)):
        for j in range(i + 1, len(atoms_3d)):
            dist = dists[i, j]
            # 我们只关心成键范围内的距离
            if dist < max_bond:
                # 如果一个非常短的距离出现了，这通常意味着优化失败或结构不合理
                if dist < min_bond:
                    print(f"[Reject] Unreasonable bond length detected: {dist:.2f} Å")
                    return False

    return True

# =============================================================================
# PART 5: MAIN WORKFLOW
# =============================================================================

def run_one(i, total, seed):
    """包装单分子生成逻辑，使其可被多进程调用"""
    try:
        rng = random.Random(seed)
        print(f"\n[Worker {os.getpid()}] --- Generating molecule {i + 1}/{total} ---")
        symmetries = ['C2', 'C3', 'C6', 'mirror', 'asymmetric']
        target_symmetry = rng.choice(symmetries)
        num_rings = rng.randint(10, 80)
        print(f"Target symmetry: {target_symmetry}, rings: ~{num_rings}")

        centers = grow_random_symmetric(steps=num_rings, symmetry=target_symmetry, rng=rng)
        if not centers:
            print("Failed to generate centers. Skipping.")
            return False

        flake_graph, axial_map = centers_to_graph(centers)
        initial_symmetry = detect_highest_symmetry(flake_graph, axial_map)

        try:
            flake_mol_2d = graph_with_hetero_to_mol(flake_graph)
            if flake_mol_2d is None or flake_mol_2d.GetNumAtoms() == 0:
                print("Failed to generate flake RDKit molecule. Skipping.")
                return False
        except Exception as e:
            print(f"Error generating flake molecule: {e}")
            return False

        hash_prefix = get_canonical_smiles_hash(flake_mol_2d)
        print(f"Initial actual symmetry: {initial_symmetry}. Hash prefix: {hash_prefix}")

        out_path = os.path.join("symmetric_molecules", initial_symmetry)
        os.makedirs(out_path, exist_ok=True)
        save_graph_svg(flake_graph, os.path.join(out_path, f"{hash_prefix}_flake.svg"))
        save_graph_xyz(flake_graph, os.path.join(out_path, f"{hash_prefix}_flake.xyz"))

        defective_graph = apply_symmetric_sw(flake_graph, initial_symmetry, axial_map, rng=rng)
        final_graph = substitute_symmetric_atoms(defective_graph, initial_symmetry, axial_map, elem='N', num_sites=1, rng=rng)
        final_graph = substitute_symmetric_atoms(final_graph, initial_symmetry, axial_map, elem='O', num_sites=1, rng=rng)
        final_graph = substitute_symmetric_atoms(final_graph, initial_symmetry, axial_map, elem='B', num_sites=1, rng=rng)

        final_symmetry_actual = detect_highest_symmetry(final_graph, axial_map)
        out_path_final = os.path.join("symmetric_molecules", final_symmetry_actual)
        os.makedirs(out_path_final, exist_ok=True)
        print(f"Final actual symmetry: {final_symmetry_actual}")

        final_mol_2d = graph_with_hetero_to_mol(final_graph)
        if final_mol_2d is None or final_mol_2d.GetNumAtoms() == 0:
            print("未能生成最终的RDKit分子，跳过。")
            return False

        final_mol_2d = add_internal_bonds(final_mol_2d)
        final_ase_atoms = get_structure_from_mol2d(final_mol_2d)
        if final_ase_atoms is None:
            print("因3D嵌入失败，跳过保存此分子。")
            return False

        save_mol_svg(final_mol_2d, os.path.join(out_path_final, f"{hash_prefix}_final.svg"))
        save_ase_xyz(final_ase_atoms, os.path.join(out_path_final, f"{hash_prefix}_final.xyz"),
                     comment=f"Final molecule {i + 1}/{total}")
        print(f"成功为 {hash_prefix} 保存文件于 '{out_path_final}'")
        return True

    except Exception as e:
        print(f"[Worker {os.getpid()}] Unhandled exception: {e}")
        traceback.print_exc()
        return False

def main():
    N_MOLECULES = 10
    pool_size = min(os.cpu_count() or 1, N_MOLECULES)
    print(f"Using pool size: {pool_size}")

    # 为每个任务分配不同随机种子
    seeds = [random.randrange(1, 2**30) for _ in range(N_MOLECULES)]
    args = [(i, N_MOLECULES, seeds[i]) for i in range(N_MOLECULES)]

    # 启动进程池
    with mp.Pool(processes=pool_size) as pool:
        results = pool.starmap(run_one, args)

    # 汇总结果
    success = sum(1 for r in results if r)
    print(f"\n✅ Completed: {success}/{N_MOLECULES} molecules succeeded.")

if __name__ == "__main__":
    main()