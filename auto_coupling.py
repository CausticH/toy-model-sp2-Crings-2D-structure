import os, csv, random, hashlib, numpy as np, datetime, sys
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem import rdmolfiles

# ======== user knobs ========
IN_DIR        = "symmetric_molecules_small"   # root folder, will auto-scan all subfolders
TARGET_COUNT  = 1000                     # how many products to make
OUT_DIR       = "coupling"              # output folder
PAUSE_AT_END  = True                    # pause console on exit (for double-click)
# ============================


# ------------ Molecule helpers ------------
def load_mol(path):
    mol = Chem.MolFromMolFile(path, sanitize=False)
    return mol
def mol_hash16(m):
    """Canonical SMILES → sha256 → 16 hex."""
    smi = Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)
    return hashlib.sha256(smi.encode()).hexdigest()[:16]
def get_canonical_ranks(mol):
    """
    Canonical atom 'symmetry classes' using RDKit's CanonicalRankAtoms.
    breakTies=False
    """
    return list(rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
def find_edge_secondary_sites(mol):
    """
    Return a list of edge secondary C–H sites suitable for coupling.
    Each item: (c_idx, h_idx), where carbon has degree==2 and has an attached H.
    """
    res = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 6:
            continue
        neigh = a.GetNeighbors()
        heavy = [x for x in neigh if x.GetAtomicNum() != 1]
        Hs = [x for x in neigh if x.GetAtomicNum() == 1]
        if len(heavy) == 2 and len(Hs) >= 1:
            res.append((a.GetIdx(), Hs[0].GetIdx()))
    return res

# ------------edge detection & simple coupling ------------
def edge_sites(m):
    """
    Return list of (c_idx, h_idx)
    where c is carbon with exactly 2 heavy neighbors + ≥1 H neighbor
    """
    out = []
    for a in m.GetAtoms():
        if a.GetSymbol() != "C":
            continue

        neigh = a.GetNeighbors()
        heavy = [x for x in neigh if x.GetSymbol() != "H"]
        Hs    = [x for x in neigh if x.GetSymbol() == "H"]

        if len(heavy) == 2 and len(Hs) >= 1:
            out.append((a.GetIdx(), Hs[0].GetIdx()))
    return out

def couple_simple(m1, m2):
    """
    Remove one terminal H from each → C–C single bond
    Then 3D embed + UFF optimize
    """
    m1 = Chem.RWMol(m1)
    m2 = Chem.RWMol(m2)

    s1 = edge_sites(m1)
    s2 = edge_sites(m2)
    if not s1 or not s2:
        raise ValueError("No suitable edge H sites found")

    c1, h1 = random.choice(s1)
    c2, h2 = random.choice(s2)

    # remove hydrogens
    m1.RemoveAtom(h1)
    m2.RemoveAtom(h2)

    combo = Chem.CombineMols(m1, m2)
    rw = Chem.RWMol(combo)

    off = m1.GetNumAtoms()
    rw.AddBond(c1, off + c2, Chem.BondType.SINGLE)

    mol = Chem.AddHs(rw)
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    # =================================

    return mol
def self_couple(mol, mode=None):
    """
    Self-coupling by SAME-RANK pairing between original molecule and its copy.
    mode: "C2", "mirror", "asymmetric"
    """
    mol1 = Chem.Mol(mol)
    mol2 = Chem.Mol(mol)

    ranks1 = get_canonical_ranks(mol1)
    ranks2 = get_canonical_ranks(mol2)

    sites1 = find_edge_secondary_sites(mol1)  # [(c,h),...]
    sites2 = find_edge_secondary_sites(mol2)

    by_rank_1 = {}
    for (c, h) in sites1:
        r = ranks1[c]
        by_rank_1.setdefault(r, []).append((c, h))

    by_rank_2 = {}
    for (c, h) in sites2:
        r = ranks2[c]
        by_rank_2.setdefault(r, []).append((c, h))

    chosen = None

    if mode in ("C2", "mirror"):
        for r in (set(by_rank_1.keys()) & set(by_rank_2.keys())):
            if by_rank_1[r] and by_rank_2[r]:
                chosen = (r, by_rank_1[r][0], by_rank_2[r][0])
                break
        if chosen is None:
            raise ValueError("self_couple: no SAME-RANK edge-CH for C2/mirror")

    else:
        # get any edge-CH
        if sites1 and sites2:
            (c1, h1) = random.choice(sites1)
            (c2, h2) = random.choice(sites2)
            if (c1,h1)!=(c2,h2):
                chosen = ("asym", (c1, h1), (c2, h2))
            else:
                raise ValueError("self_couple: no edge-CH available for asymmetric")
        else:
            raise ValueError("self_couple: no edge-CH available for asymmetric")

    # unpack
    _, (c1, h1), (c2, h2) = chosen

    r, (c1, h1), (c2, h2) = chosen

    combo = Chem.CombineMols(mol1, mol2)
    rw = Chem.RWMol(combo)

    offset = mol1.GetNumAtoms()
    c2_w = c2 + offset  # partner atom in second copy
    h2_w = h2 + offset  # partner H in second copy

    # Mark the two key C atoms so we can re-find them after embedding
    rw.GetAtomWithIdx(c1).SetIntProp("_tag_sc", 1)
    rw.GetAtomWithIdx(c2_w).SetIntProp("_tag_sc", 2)

    # --- Remove the two H atoms BEFORE embedding ---
    # remove in reverse order by index
    for h in sorted([h1, h2_w], reverse=True):
        rw.RemoveAtom(h)

    mol3 = rw.GetMol()

    # --- Embed heavy-atom skeleton ---
    ok = AllChem.EmbedMolecule(mol3, AllChem.ETKDG())
    if ok != 0:
        ok = AllChem.EmbedMolecule(mol3, useRandomCoords=True)
        if ok != 0:
            raise ValueError("self_couple: Embed failed")

    # --- Re-locate tagged atoms (C1 from block-1, C2 from block-2) ---
    new_c1 = new_c2 = None
    for a in mol3.GetAtoms():
        if a.HasProp("_tag_sc"):
            t = a.GetIntProp("_tag_sc")
            if t == 1:
                new_c1 = a.GetIdx()
            elif t == 2:
                new_c2 = a.GetIdx()
    if new_c1 is None or new_c2 is None:
        raise RuntimeError("self_couple: lost tags after embed")

    # --- Apply transform to *ALL* atoms in second copy ---
    second_copy_atoms = list(range(offset, mol3.GetNumAtoms()))

    conf = mol3.GetConformer()
    p1 = np.array(conf.GetAtomPosition(new_c1))
    p2 = np.array(conf.GetAtomPosition(new_c2))

    # direction vector: new_c1 → new_c2
    v = p2 - p1
    norm = np.linalg.norm(v)
    if norm < 1e-4:
        # fallback direction
        v = np.array([1.0, 0.0, 0.0])
        norm = 1.0
    v /= norm

    shift = 1.42 * v

    # shift all atoms belonging to the second copy
    for i in second_copy_atoms:
        pi = np.array(conf.GetAtomPosition(i))
        conf.SetAtomPosition(i, pi + shift)

    # --- Add C–C bond between new_c1 and new_c2 ---
    if mol3.GetBondBetweenAtoms(new_c1, new_c2) is None:
        rw2 = Chem.RWMol(mol3)
        rw2.AddBond(new_c1, new_c2, Chem.BondType.SINGLE)
        mol4 = rw2.GetMol()
    else:
        mol4 = mol3

    mol_tmp = Chem.RemoveHs(mol4)
    AllChem.EmbedMolecule(mol_tmp, AllChem.ETKDG())
    mol_tmp = Chem.AddHs(mol_tmp)
    AllChem.UFFOptimizeMolecule(mol_tmp)

    return Chem.RemoveHs(mol_tmp)

# ------------ Runner ------------
def auto_run(in_dir, target_count, out_dir="coupling"):
    print("=== Coupling started ===")
    print(f"Input root: {os.path.abspath(in_dir)}")
    print(f"Target products: {target_count}")
    print(f"Output dir: {os.path.abspath(out_dir)}")

    # ---- NEW: auto scan all subfolders ----
    precs = []
    subdirs = []

    root_dir = "coupling_once"
    self_root = os.path.join(root_dir, "self_coupling")

    self_dir_sym = os.path.join(self_root, "symmetric")
    self_dir_as = os.path.join(self_root, "asymmetric")

    diff_dir = os.path.join(root_dir, "different_coupling")

    for _d in (self_dir_sym, self_dir_as, diff_dir):
        os.makedirs(_d, exist_ok=True)

    dbfile = os.path.join(root_dir, "coupling_once.csv")
    if not os.path.exists(dbfile):
        with open(dbfile, "w", newline="") as f:
            csv.writer(f).writerow(
                ["coupling_mode", "sym1", "hash1", "sym2", "hash2", "new_hash"]
            )

    for sub in os.listdir(in_dir):
        print("SCAN SUBDIR:", sub)
        d = os.path.join(in_dir, sub)
        if not os.path.isdir(d):
            continue
        subdirs.append(sub)

        files = [f for f in os.listdir(d) if f.lower().endswith(".mol")]
        print(f"  - {sub}: {len(files)} files")

        for fn in files:
            path = os.path.join(d, fn)
            try:
                m = load_mol(path)
                h = mol_hash16(m)
                precs.append((m, sub, h))
            except Exception:
                pass

    print(f"Total precursors loaded: {len(precs)}")
    if len(precs) < 2:
        print("Not enough precursor molecules. Abort.")
        return

        # collect existing hashes from new folders
    existing = set()
    scan_dirs = [diff_dir, self_dir_sym, self_dir_as]
    for _sd in scan_dirs:
        for x in os.listdir(_sd):
            if x.lower().endswith(".mol") and len(x.split(".")[0]) >= 16:
                existing.add(x.split(".")[0])

    cnt = len(existing)
    print(f"Existing products found in coupling_once tree: {cnt}")

    while cnt < target_count:
        try:
            if random.random() < 0.5:
                print(">>> ENTER SELF MODE")
                # ---------- self-coupling ----------
                (m1, s1, h1) = random.choice(precs)

                mode = random.choice(["C2", "mirror", "asymmetric"])
                coupling_mode = {"C2": "self_sym", "mirror": "self_sym", "asymmetric": "self_as"}[mode]

                new = self_couple(Chem.Mol(m1), mode=mode)
                new_h = mol_hash16(new)
                if new_h in existing:
                    print(f"[DUP] {new_h}, retry another pair (self)")
                    continue

                if mode in ("C2", "mirror"):
                    out_sub = self_dir_sym
                else:
                    out_sub = self_dir_as

                s2, h2 = s1, h1

            else:
                # ---------- different-coupling ----------
                (m1, s1, h1), (m2, s2, h2) = random.sample(precs, 2)
                coupling_mode = "diff"

                new = couple_simple(Chem.Mol(m1), Chem.Mol(m2))
                new_h = mol_hash16(new)
                if new_h in existing:
                    print(f"[DUP] {new_h}, retry another pair (diff)")
                    continue

                out_sub = diff_dir

            prefix = os.path.join(out_sub, new_h)
            Chem.MolToMolFile(new, prefix + ".mol")
            with open(prefix + ".xyz", "w") as f:
                f.write(Chem.MolToXYZBlock(new))

            with open(dbfile, "a", newline="") as f:
                csv.writer(f).writerow(
                    [coupling_mode, s1, h1, s2, h2, new_h]
                )

            existing.add(new_h)
            cnt += 1
            print(f"[OK] {new_h}  ({cnt}/{target_count})  mode={coupling_mode}")

        except Exception as e:
            print(f"[SKIP] ERROR: {e}")

    print(f"Done → {cnt} molecules → {out_dir}/")


if __name__ == "__main__":
    auto_run(IN_DIR, TARGET_COUNT, OUT_DIR)
    if PAUSE_AT_END:
        try:
            input("\nPress Enter to exit...")
        except EOFError:
            pass