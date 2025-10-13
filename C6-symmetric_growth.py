from rdkit import Chem
from rdkit.Chem import rdDepictor, Draw, rdmolops
import math, os, hashlib, random, multiprocessing, time, itertools


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
    """根据中心坐标生成分子"""
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
    注意：C4 对称性与六边形晶格不兼容。
    """
    q, r = pos
    if symmetry == 'C6':
        # 60度旋转6次
        p = (q, r)
        equivalents = set()
        for _ in range(6):
            equivalents.add(p)
            p = (p[0] + p[1], -p[0])  # C6 旋转矩阵
        return list(equivalents)
    elif symmetry == 'C3':
        p = (q, r)
        equivalents = set()
        for _ in range(3):
            equivalents.add(p)
            p = (-p[0] - p[1], p[0])
        return list(equivalents)
    elif symmetry == 'C2':
        # 180度旋转
        return [(q, r), (-q, -r)]
    elif symmetry == 'mirror':
        # 沿 q+r=0 轴镜面反射
        return [(q, r), (-r, -q)]
    else:  # C1 (asymmetric)
        return [(q, r)]


def detect_highest_symmetry(centers):
    """
    检测给定中心点集合的最高阶对称性。
    优先级: C6 > C3 > C2 > Mirror > Asymmetric (C1)
    """
    center_set = set(centers)
    # 检查 C6
    if all((q + r, -q) in center_set for q, r in center_set):
        return 'C6'
    # 检查 C3
    if all((-q - r, q) in center_set for q, r in center_set):
        return 'C3'
    # 检查 C2
    if all((-q, -r) in center_set for q, r in center_set):
        return 'C2'
    # 检查镜面
    if all((-r, -q) in center_set for q, r in center_set):
        return 'mirror'
    return 'asymmetric'


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
        seed_centers = [(0, 0)]  # 对称性生长必须从中心开始

    centers = set(seed_centers)
    # 初始点也必须符合对称性
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

        # 添加新点及其所有对称等效点
        new_points = get_symmetric_equivalents(new, symmetry)

        for p in new_points:
            centers.add(p)

        if len(centers) == current_rings:  # 无法继续生长
            break
        current_rings = len(centers)

    return sorted(list(centers))


# ==========================================================
#                    筛选与检测函数
# ==========================================================

def add_internal_bonds(mol, max_dist=1.55):
    """
    优先级: 6元环 > 5元环 > 其他环 > 更短距离
    """
    while True:
        bonds_added_in_this_iteration = False
        rw_mol = Chem.RWMol(mol)
        if rw_mol.GetNumConformers() == 0:
            Chem.rdDepictor.Compute2DCoords(rw_mol)
        conf = rw_mol.GetConformer()

        degree2_atoms_indices = [a.GetIdx() for a in rw_mol.GetAtoms() if a.GetDegree() == 2]

        if len(degree2_atoms_indices) < 2:
            break

        potential_bonds = []
        for i, j in itertools.combinations(degree2_atoms_indices, 2):
            if rw_mol.GetBondBetweenAtoms(i, j) is None:
                dist = conf.GetAtomPosition(i).Distance(conf.GetAtomPosition(j))
                if dist < max_dist:
                    path = rdmolops.GetShortestPath(rw_mol, i, j)
                    if not path: continue
                    new_ring_size = len(path)
                    potential_bonds.append({'atoms': (i, j), 'dist': dist, 'ring_size': new_ring_size})

        if not potential_bonds:
            break

        def sort_key(bond_info):
            size = bond_info['ring_size']
            dist = bond_info['dist']
            if size == 6:
                priority = 0
            elif size == 5:
                priority = 1
            else:
                priority = 2
            return (priority, dist)

        potential_bonds.sort(key=sort_key)

        for bond_info in potential_bonds:
            i, j = bond_info['atoms']
            temp_mol = Chem.RWMol(rw_mol)
            temp_mol.AddBond(i, j, Chem.BondType.SINGLE)

            try:
                Chem.SanitizeMol(temp_mol)
                Chem.SetAromaticity(temp_mol)
                mol = temp_mol.GetMol()
                bonds_added_in_this_iteration = True
                break
            except Exception:
                continue

        if not bonds_added_in_this_iteration:
            break

    final_mol = mol
    if final_mol.GetNumConformers() > 0:
        Chem.rdDepictor.Compute2DCoords(final_mol)
    return final_mol


def is_valid_structure(mol, centers, size=1.0, tol=0.15, atom_coords_override=None):
    """
    检查每个中心点及其 6 个对称点是否与任一原子坐标距离 < tol。
    仅检测边缘六边形。
    """
    tol2 = tol * tol

    if atom_coords_override is not None:
        atom_coords = list(atom_coords_override)
    else:
        if mol.GetNumConformers() == 0:
            Chem.rdDepictor.Compute2DCoords(mol)
        conf = mol.GetConformer()
        atom_coords = [(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y)
                       for i in range(mol.GetNumAtoms())]

    nbr_dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    center_set = set(centers)
    edge_centers = [(q, r) for (q, r) in centers if any((q + dq, r + dr) not in center_set for dq, dr in nbr_dirs)]

    check_points = set()
    for q, r in edge_centers:
        cx, cy = axial_to_cartesian(q, r, size)
        check_points.add((round(cx, 6), round(cy, 6)))
        for i in range(6):
            angle = math.radians(60 * i)
            px = cx + (size / 2.0) * math.cos(angle)
            py = cy + (size / 2.0) * math.sin(angle)
            check_points.add((round(px, 6), round(py, 6)))

    for px, py in check_points:
        for ax, ay in atom_coords:
            dx = ax - px
            dy = ay - py
            if dx * dx + dy * dy < tol2:
                return False
    return True


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
        dx, dy, dz = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        if dist < min_len or dist > max_len:
            return True
    return False


# ==========================================================
#                    导出工具
# ==========================================================

def mol_to_xyz(mol):
    conf = mol.GetConformer()
    natoms = mol.GetNumAtoms()
    lines = [str(natoms), "Generated by 2D_hex_grid_PAH_builder"]
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        lines.append(f"{atom.GetSymbol():2s} {pos.x: .6f} {pos.y: .6f} {pos.z: .6f}")
    return "\n".join(lines)


# ==========================================================
#                   主生成逻辑
# ==========================================================

def generate_random_pah_in_range(min_rings=2, max_rings=80, max_attempts=8000, rng=None):
    """随机生成环数在指定范围内的、具有特定对称性的分子"""
    if rng is None:
        rng = random.Random()

    # 随机选择一个对称性进行生长
    symmetries = ['C2', 'C3', 'C6', 'mirror', 'asymmetric']
    target_symmetry = rng.choice(symmetries)

    for attempt in range(max_attempts):
        weights = [n ** 1.3 for n in range(min_rings, max_rings + 1)]
        target_n = random.choices(range(min_rings, max_rings + 1), weights=weights)[0]

        # 调用对称生长函数
        centers = grow_random_symmetric(steps=target_n - 1, symmetry=target_symmetry, rng=rng)

        n_rings = len(centers)
        if not (min_rings <= n_rings <= max_rings):
            continue
        try:
            mol, _ = build_pah_from_centers(centers)
            Chem.rdDepictor.Compute2DCoords(mol)
            if not is_valid_structure(mol, centers, size=1.0, tol=0.15):
                continue
            if has_abnormal_bond_lengths(mol):
                continue
            smi = Chem.MolToSmiles(mol, canonical=True)

            # 生成后，检测其最终的最高对称性用于分类
            final_symmetry = detect_highest_symmetry(centers)

            return mol, smi, centers, final_symmetry
        except Exception:
            continue
    raise RuntimeError(f"Failed to generate valid PAH after {max_attempts} attempts")


# ==========================================================
#             并行 worker + 主函数
# ==========================================================

def worker_generate_once(args):
    """子进程调用单个随机生成任务"""
    min_rings, max_rings = args
    try:
        rng = random.Random()
        mol, smi, centers, symmetry = generate_random_pah_in_range(min_rings=min_rings, max_rings=max_rings, rng=rng)
        return (smi, centers, symmetry)
    except Exception:
        return None

def _wait_for_ready(async_list, timeout):
    """兼容版：返回 (finished_list, still_pending_list)"""
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
        time.sleep(0.001)   # 1 ms 轮询，CPU 占用可忽略


def demo_with_export_range(n_molecules=10, outdir="output",
                           min_rings=2, max_rings=80, n_procs=None,
                           poll_interval=0.05):
    from rdkit.Chem import Draw

    symmetry_dirs = ['C6', 'C3', 'C2', 'mirror', 'asymmetric']
    for d in symmetry_dirs:
        os.makedirs(os.path.join(outdir, d), exist_ok=True)

    n_procs = n_procs or (multiprocessing.cpu_count() - 1)
    window  = max(8, 4 * n_procs)

    pool = multiprocessing.Pool(processes=n_procs)
    seen_hashes = set()
    seen_smiles = set()
    generated = 0
    attempts  = 0
    dup_streak = 0

    pending = [pool.apply_async(worker_generate_once, ((min_rings, max_rings),))
               for _ in range(window)]

    try:
        while generated < n_molecules:
            finished, pending = _wait_for_ready(pending, poll_interval)

            for res in finished:
                attempts += 1
                tup = res.get()          # 保证 ready，不会阻塞
                # 补任务：先补后处理，确保窗口长度恒定
                pending.append(pool.apply_async(worker_generate_once, ((min_rings, max_rings),)))

                if tup is None:          # worker 异常
                    continue

                _, centers, symmetry = tup
                try:
                    mol, _ = build_pah_from_centers(centers)
                    Chem.rdDepictor.Compute2DCoords(mol)
                    mol = add_internal_bonds(mol)
                    cano_smi = Chem.MolToSmiles(mol, canonical=True)
                    h64 = hashlib.sha256(cano_smi.encode()).hexdigest()[:64]
                except Exception:
                    continue

                if h64 in seen_hashes or cano_smi in seen_smiles:
                    dup_streak += 1
                    if dup_streak >= 10 * window:   # 逃生舱
                        print("\n[ESC] 连续大量重复，提前终止")
                        break
                    continue

                dup_streak = 0
                if has_abnormal_bond_lengths(mol):
                    continue

                final_outdir = os.path.join(outdir, symmetry)
                base = os.path.join(final_outdir, h64)
                Draw.MolToFile(mol, base + ".svg", size=(400, 400))
                with open(base + ".xyz", "w") as f:
                    f.write(mol_to_xyz(mol))

                seen_hashes.add(h64)
                seen_smiles.add(cano_smi)
                generated += 1
                print(f"[OK] {generated:03d}: rings={len(centers):02d}  hash={h64}  sym={symmetry}")

                if generated >= n_molecules:
                    break
            if dup_streak >= 10 * window:
                break
    finally:
        pool.terminate()
        pool.join()

    print(f"\n成功生成 {generated} 个分子（共尝试 {attempts} 次）")


# ==========================================================
#                      启动入口
# ==========================================================

if __name__ == "__main__":
    # 生成1000个对称/非对称分子，并分类保存
    demo_with_export_range(n_molecules=1000, min_rings=6, max_rings=100)