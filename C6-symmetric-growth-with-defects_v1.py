from rdkit import Chem
from rdkit.Chem import rdDepictor, Draw, rdmolops
from rdkit.Geometry import Point3D
import math, os, hashlib, random, multiprocessing, time, itertools, collections


# ==========================================================
#                 基本几何帮助函数
# ==========================================================

def axial_to_cartesian(q, r, size=1.0):
    """把六边形的轴坐标 (q, r) 转换为笛卡尔坐标（pointy-top orientation）"""
    x = size * math.sqrt(3) * (q + r / 2.0)
    y = size * 3.0 / 2.0 * r
    return (x, y)


def hex_corner(center, size, i):
    """返回第 i 个角的坐标（i=0..5），使用 pointy-top orientation。"""
    (cx, cy) = center
    angle_deg = 60 * i + 30
    angle_rad = math.radians(angle_deg)
    return (cx + size * math.cos(angle_rad), cy + size * math.sin(angle_rad))


# ==========================================================
#                 构建拓扑（顶点去重 & 键连接）
# ==========================================================

def build_pah_from_centers(centers, size=1.0, coord_decimals=6):
    """
    根据中心坐标生成分子，并将完美的几何坐标作为构象存入分子对象。
    """
    vertex_map = {}
    vertex_coord_exact = {}
    bonds = set()
    rw = Chem.RWMol()

    def round_coord(pt):
        return (round(pt[0], coord_decimals), round(pt[1], coord_decimals))

    for center_ax in centers:
        center = axial_to_cartesian(center_ax[0], center_ax[1], size=size)
        verts = [hex_corner(center, size, i) for i in range(6)]

        idxs = []
        for v in verts:
            rv = round_coord(v)
            if rv in vertex_map:
                idx = vertex_map[rv]
            else:
                a = Chem.Atom(6)
                idx = rw.AddAtom(a)
                vertex_map[rv] = idx
                vertex_coord_exact[idx] = v
            idxs.append(idx)

        for i in range(6):
            a1, a2 = idxs[i], idxs[(i + 1) % 6]
            bond_pair = frozenset({a1, a2})
            if bond_pair not in bonds:
                bonds.add(bond_pair)
                rw.AddBond(a1, a2, Chem.BondType.SINGLE)

    mol = rw.GetMol()

    if mol.GetNumAtoms() > 0:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for idx, pos_tuple in vertex_coord_exact.items():
            conf.SetAtomPosition(idx, Point3D(pos_tuple[0], pos_tuple[1], 0.0))
        mol.AddConformer(conf, assignId=True)

    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
    Chem.SetAromaticity(mol)

    return mol, vertex_coord_exact


# ==========================================================
#                   对称性操作
# ==========================================================

def get_symmetric_equivalents(pos, symmetry):
    """
    根据给定的对称性，返回一个坐标的所有对称等效点。
    坐标系为轴坐标 (q, r)。
    """
    q, r = pos
    if symmetry == 'C6':
        p = (q, r)
        equivalents = set()
        for _ in range(6):
            equivalents.add(p)
            p = (p[0] + p[1], -p[0])
        return list(equivalents)
    elif symmetry == 'C3':
        p = (q, r)
        equivalents = set()
        for _ in range(3):
            equivalents.add(p)
            p = (-p[0] - p[1], p[0])
        return list(equivalents)
    elif symmetry == 'C2':
        return [(q, r), (-q, -r)]


def detect_highest_symmetry(centers):
    """
    检测给定中心点集合的最高阶对称性。
    """
    center_set = set(centers)
    if all((q + r, -q) in center_set for q, r in center_set):
        return 'C6'
    if all((-q - r, q) in center_set for q, r in center_set):
        return 'C3'
    if all((-q, -r) in center_set for q, r in center_set):
        return 'C2'


def _map_real_to_perfect(mol, perfect_keys, coord_decimals=6):
    """
    把真实原子坐标映射到最近的 perfect 坐标
    """

    def round_coord(pt):
        return (round(pt[0], coord_decimals), round(pt[1], coord_decimals))

    conf = mol.GetConformer()
    mapping = {k: [] for k in perfect_keys}
    for idx in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6:
            continue
        x, y = conf.GetAtomPosition(idx).x, conf.GetAtomPosition(idx).y
        key = round_coord((x, y))
        if key in mapping:
            mapping[key].append(idx)
    return mapping


# ==========================================================
#                     随机扩展逻辑
# ==========================================================

def grow_random_symmetric(steps, symmetry, seed_centers=None, rng=None):
    """
    按指定的对称性进行随机生长。
    """
    if rng is None:
        rng = random.Random()
    if seed_centers is None:
        seed_centers = [(0, 0)]

    centers = set(seed_centers)
    initial_equivalents = get_symmetric_equivalents(seed_centers[0], symmetry)
    for p in initial_equivalents:
        centers.add(p)

    nbr_dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    current_rings = len(centers)
    while current_rings < steps + 1:
        candidates = set()
        for c in centers:
            for d in nbr_dirs:
                pos = (c[0] + d[0], c[1] + d[1])
                if pos not in centers:
                    candidates.add(pos)

        if not candidates:
            break

        new = rng.choice(list(candidates))
        new_points = get_symmetric_equivalents(new, symmetry)

        for p in new_points:
            centers.add(p)

        if len(centers) == current_rings:
            break
        current_rings = len(centers)

    return sorted(list(centers))

# ==========================================================
#                    筛选与检测函数
# ==========================================================

def add_internal_bonds(mol, max_dist=1.55):
    """
    一次性闭环
    """
    rw_mol = Chem.RWMol(mol)
    if rw_mol.GetNumConformers() == 0:
        Chem.rdDepictor.Compute2DCoords(rw_mol)
    conf = rw_mol.GetConformer()

    # 1) 先一次性收集所有潜在内键
    degree2 = {a.GetIdx() for a in rw_mol.GetAtoms() if a.GetDegree() == 2}
    potential = []
    for i, j in itertools.combinations(degree2, 2):
        if rw_mol.GetBondBetweenAtoms(i, j) is not None:
            continue
        dist = conf.GetAtomPosition(i).Distance(conf.GetAtomPosition(j))
        if dist >= max_dist:
            continue
        path = rdmolops.GetShortestPath(rw_mol, i, j)
        if not path:
            continue
        potential.append({'atoms': (i, j), 'ring_size': len(path), 'dist': dist})

    # 2) 按优先级排序
    potential.sort(key=lambda b: (0 if b['ring_size'] == 6 else 1 if b['ring_size'] == 5 else 2, b['dist']))

    # 3) 一次性加键（不加一轮、不加第二轮）
    for b in potential:
        i, j = b['atoms']
        rw_mol.AddBond(i, j, Chem.BondType.SINGLE)

    # 4) 最后一次性 sanitize
    try:
        Chem.SanitizeMol(rw_mol)
        Chem.SetAromaticity(rw_mol)
    except Exception:
        return mol

    final = rw_mol.GetMol()
    if final.GetNumConformers() == 0:
        Chem.rdDepictor.Compute2DCoords(final)
    return final

# ==========================================================
#                    缺陷与检测函数
# ==========================================================

def _generate_perfect_vertex_to_idx(centers, coord_decimals=6):
    """生成 perfect 顶点坐标池"""

    def round_coord(pt):
        return (round(pt[0], coord_decimals), round(pt[1], coord_decimals))

    perfect = {}
    for q, r in centers:
        cx, cy = axial_to_cartesian(q, r)
        for i in range(6):
            x, y = hex_corner((cx, cy), 1.0, i)
            key = round_coord((x, y))
            perfect.setdefault(key, []).append(None)
    return perfect


def _get_all_center_layers(centers):
    """
    使用广度优先搜索（BFS）正确计算所有六边形中心点距离边缘的层数。
    """
    center_set = set(centers)
    if not center_set:
        return {}

    nbr_dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    layers = {}
    queue = collections.deque()

    for q, r in centers:
        is_edge = False
        for dq, dr in nbr_dirs:
            if (q + dq, r + dr) not in center_set:
                is_edge = True
                break
        if is_edge:
            layers[(q, r)] = 0
            queue.append((q, r))

    visited = set(queue)

    while queue:
        (cq, cr) = queue.popleft()
        current_layer = layers.get((cq, cr), 0)

        for dq, dr in nbr_dirs:
            nq, nr = cq + dq, cr + dr
            if (nq, nr) in center_set and (nq, nr) not in visited:
                layers[(nq, nr)] = current_layer + 1
                visited.add((nq, nr))
                queue.append((nq, nr))

    for c in centers:
        if c not in layers:
            layers[c] = 0

    return layers


def defect_radius_delete(mol, centers, symmetry, rng=None):
    """
    任何缺陷必须产生于第2层或更深, 无法生成则返回 None。
    """
    if rng is None:
        rng = random.Random()
    if mol.GetNumAtoms() == 0:
        return None

    rw = Chem.RWMol(mol)

    all_layers = _get_all_center_layers(centers)
    if not all_layers: return None
    max_layer = max(all_layers.values()) if all_layers else 0

    if max_layer < 2:
        return None

    target_layer = rng.randint(2, max_layer)
    candidate_centers = [c for c, layer in all_layers.items() if layer == target_layer]
    if not candidate_centers: return None

    center_groups, processed_centers = [], set()
    for center in candidate_centers:
        if center in processed_centers: continue
        equiv_group = get_symmetric_equivalents(center, symmetry)
        group_to_add = [eq_c for eq_c in equiv_group if eq_c in candidate_centers]
        if group_to_add:
            center_groups.append(group_to_add)
            processed_centers.update(group_to_add)

    if not center_groups: return None

    doomed_centers = rng.choice(center_groups)

    coord_decimals = 6

    def round_coord(pt):
        return (round(pt[0], coord_decimals), round(pt[1], coord_decimals))

    doomed_coords = set()
    for q, r in doomed_centers:
        center_cart = axial_to_cartesian(q, r, size=1.0)
        for i in range(6):
            v = hex_corner(center_cart, 1.0, i)
            doomed_coords.add(round_coord(v))

    perfect_keys = list(_generate_perfect_vertex_to_idx(centers).keys())
    real_to_perfect = _map_real_to_perfect(rw, perfect_keys)

    doomed_indices = set()
    for pk, idx_list in real_to_perfect.items():
        if pk in doomed_coords:
            doomed_indices.update(idx_list)

    if not doomed_indices: return None

    for idx in sorted(list(doomed_indices), reverse=True):
        rw.RemoveAtom(idx)

    changed = True
    while changed:
        changed = False
        for atm in list(rw.GetAtoms()):
            if atm.GetDegree() == 1:
                rw.RemoveAtom(atm.GetIdx())
                changed = True

    new_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(new_mol,
                         sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                                     Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    except Exception:
        return None

    return new_mol

# 清理函数
def cleanup_fragments(mol):
    """
    将分子拆分为独立的碎片，并只返回其中最大的（原子数最多）一个。
    用于去除缺陷生成后可能产生的孤立小碎片。
    """
    # asMols=True可以直接得到Mol对象列表
    fragments = rdmolops.GetMolFrags(mol, asMols=True)

    # 如果分子本身就是相连的（只有一个碎片），则无需清理
    if len(fragments) <= 1:
        return mol

    # 找到最大的碎片
    largest_frag = None
    max_atoms = 0
    for frag in fragments:
        num_atoms = frag.GetNumAtoms()
        if num_atoms > max_atoms:
            max_atoms = num_atoms
            largest_frag = frag

    return largest_frag


def has_abnormal_bond_lengths(mol, min_len=1.2, max_len=1.6):
    """检测是否存在异常的 C–C 键长"""
    if mol.GetNumConformers() == 0:
        Chem.rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        p1 = conf.GetAtomPosition(a1)
        p2 = conf.GetAtomPosition(a2)
        dist = p1.Distance(p2)
        if dist < min_len or dist > max_len:
            return True
    return False


def mol_to_xyz(mol):
    """将RDKit分子对象转换为XYZ格式的字符串"""
    if mol.GetNumConformers() == 0:
        return ""
    conf = mol.GetConformer()
    natoms = mol.GetNumAtoms()
    lines = [str(natoms), "Generated Molecule"]
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        lines.append(f"{atom.GetSymbol():2s} {pos.x: .6f} {pos.y: .6f} {pos.z: .6f}")
    return "\n".join(lines)


def worker_generate_once(args):
    """子进程调用单个随机生成任务"""
    min_rings, max_rings = args
    try:
        rng = random.Random()
        target_symmetry = rng.choice(['C2', 'C3', 'C6'])
        target_n = rng.randint(min_rings, max_rings)
        centers = grow_random_symmetric(steps=target_n - 1, symmetry=target_symmetry, rng=rng)
        if not (min_rings <= len(centers) <= max_rings):
            return None
        return (None, centers, target_symmetry)
    except Exception:
        return None


def _wait_for_ready(async_list, timeout):
    """返回 (finished_list, still_pending_list)"""
    t0 = time.time()
    finished = []
    while True:
        still_pending = []
        for r in async_list:
            if r.ready():
                finished.append(r)
            else:
                still_pending.append(r)
        if finished or (time.time() - t0 >= timeout):
            return finished, still_pending
        time.sleep(0.001)


def demo_with_export_range(n_molecules=10, outdir="output", min_rings=20, max_rings=100, n_procs=None,
                           poll_interval=0.05):
    from rdkit.Chem import Draw

    symmetry_dirs = ['C6', 'C3', 'C2']
    for d in symmetry_dirs:
        os.makedirs(os.path.join(outdir, d), exist_ok=True)

    n_procs = n_procs or (multiprocessing.cpu_count() - 1)
    window = max(8, 4 * n_procs)

    pool = multiprocessing.Pool(processes=n_procs)
    seen_hashes = set()
    seen_smiles = set()
    generated = 0
    attempts = 0
    dup_streak = 0

    pending = [pool.apply_async(worker_generate_once, ((min_rings, max_rings),))
               for _ in range(window)]

    try:
        while generated < n_molecules:
            finished, pending = _wait_for_ready(pending, poll_interval)

            for res in finished:
                attempts += 1
                tup = res.get()
                pending.append(pool.apply_async(worker_generate_once, ((min_rings, max_rings),)))

                if tup is None:
                    continue

                _, centers, symmetry = tup
                if not centers: continue

                try:
                    mol, _ = build_pah_from_centers(centers)

                    mol = add_internal_bonds(mol)

                    mol_with_defect = defect_radius_delete(mol, centers, symmetry, rng=random.Random())

                    if mol_with_defect is None: continue

                    mol_final = mol_with_defect

                    # ===================================================================
                    #  ↓↓↓ 调用清理函数 ↓↓↓
                    # ===================================================================
                    mol_final = cleanup_fragments(mol_final)
                    if mol_final is None: continue  # 如果清理后分子没了，也跳过

                    Chem.rdDepictor.Compute2DCoords(mol_final)

                    cano_smi = Chem.MolToSmiles(mol_final, canonical=True)
                    h64 = hashlib.sha256(cano_smi.encode()).hexdigest()[:64]
                except Exception:
                    continue

                if h64 in seen_hashes or cano_smi in seen_smiles:
                    dup_streak += 1
                    if dup_streak >= 10 * window:
                        print("\n[ESC] 连续大量重复，提前终止")
                        break
                    continue

                dup_streak = 0
                if has_abnormal_bond_lengths(mol_final):
                    continue

                final_symmetry = detect_highest_symmetry(centers)
                final_outdir = os.path.join(outdir, final_symmetry)
                os.makedirs(final_outdir, exist_ok=True)

                base = os.path.join(final_outdir, h64)
                Draw.MolToFile(mol_final, base + ".svg", size=(400, 400))
                with open(base + ".xyz", "w") as f:
                    f.write(mol_to_xyz(mol_final))

                seen_hashes.add(h64)
                seen_smiles.add(cano_smi)
                generated += 1
                print(f"[OK] {generated:03d}: rings={len(centers):02d}  hash={h64[:10]}...  sym={final_symmetry}")

                if generated >= n_molecules:
                    break
            if dup_streak >= 10 * window:
                break
    finally:
        pool.terminate()
        pool.join()

    print(f"\n成功生成 {generated} 个分子（共尝试 {attempts} 次）")


if __name__ == "__main__":
    demo_with_export_range(n_molecules=100, min_rings=20, max_rings=100)