import os, csv, random, hashlib, time, sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles

# =============== USER KNOBS ===============
IN_DIR         = "symmetric_molecules_small"   # root folder of precursors (auto-scan subfolders)
TARGET_COUNT   = 6000                           # how many products to make
ROOT_OUT       = "coupling_once"                # output root (same as before)
TIME_LIMIT_SEC = 200                            # per-task soft timeout (seconds)
MAX_INFLIGHT   = None                           # None => 2 * n_workers
PAUSE_AT_END   = True                           # pause console on exit
# =========================================

# ------------ helpers (do NOT pass RDKit Mol across processes) ------------
def mol_hash16(mol) -> str:
    """Canonical SMILES → sha256 → first 16 hex chars."""
    smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return hashlib.sha256(smi.encode()).hexdigest()[:16]

def load_molblock(path: str) -> str:
    """Read .mol as text (MolBlock)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def get_canonical_ranks(mol):
    """RDKit Canonical symmetry classes (breakTies=False)."""
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
        Hs    = [x for x in neigh if x.GetAtomicNum() == 1]
        if len(heavy) == 2 and len(Hs) >= 1:
            res.append((a.GetIdx(), Hs[0].GetIdx()))
    return res

def _edge_sites(m):
    """Return list of (c_idx, h_idx) where c has exactly 2 heavy neighbors + >=1 H neighbor."""
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

def _couple_simple_from_blocks(mb1: str, mb2: str) -> Chem.Mol:
    """
    Remove one terminal H from each → C–C single bond
    Then 3D embed + UFF optimize
    """
    m1 = Chem.MolFromMolBlock(mb1, sanitize=False)
    m2 = Chem.MolFromMolBlock(mb2, sanitize=False)

    m1 = Chem.RWMol(m1)
    m2 = Chem.RWMol(m2)

    s1 = _edge_sites(m1)
    s2 = _edge_sites(m2)
    if not s1 or not s2:
        raise ValueError("No suitable edge H sites found")

    c1, h1 = random.choice(s1)
    c2, h2 = random.choice(s2)

    # remove hydrogens (remove in descending index order is safer)
    for h in sorted([h1], reverse=True):
        m1.RemoveAtom(h)
    for h in sorted([h2], reverse=True):
        m2.RemoveAtom(h)

    combo = Chem.CombineMols(m1, m2)
    rw = Chem.RWMol(combo)

    off = m1.GetNumAtoms()
    rw.AddBond(c1, off + c2, Chem.BondType.SINGLE)

    mol = rw.GetMol()
    mol = Chem.AddHs(mol)
    # ETKDG embed + optimize; retry on failure with random coords
    ok = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if ok != 0:
        ok = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if ok != 0:
            raise ValueError("Embed failed in couple_simple")

    AllChem.UFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    return mol

def _self_couple_from_block(mb: str, mode: str) -> Chem.Mol:
    """
    Self-coupling by SAME-RANK pairing between original molecule and its copy.
    mode: "C2", "mirror", "asymmetric"
    """
    mol1 = Chem.MolFromMolBlock(mb, sanitize=False)
    mol2 = Chem.MolFromMolBlock(mb, sanitize=False)

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
        if sites1 and sites2:
            (c1, h1) = random.choice(sites1)
            (c2, h2) = random.choice(sites2)
            if (c1, h1) != (c2, h2):
                chosen = ("asym", (c1, h1), (c2, h2))
            else:
                raise ValueError("self_couple: duplicated pick for asymmetric")
        else:
            raise ValueError("self_couple: no edge-CH available for asymmetric")

    # unpack
    _, (c1, h1), (c2, h2) = chosen
    r, (c1, h1), (c2, h2) = chosen

    combo = Chem.CombineMols(mol1, mol2)
    rw = Chem.RWMol(combo)

    offset = mol1.GetNumAtoms()
    c2_w = c2 + offset
    h2_w = h2 + offset

    # mark the two carbons
    rw.GetAtomWithIdx(c1).SetIntProp("_tag_sc", 1)
    rw.GetAtomWithIdx(c2_w).SetIntProp("_tag_sc", 2)

    # remove two H first (descending index)
    for h in sorted([h1, h2_w], reverse=True):
        rw.RemoveAtom(h)

    mol3 = rw.GetMol()

    # embed skeleton
    ok = AllChem.EmbedMolecule(mol3, AllChem.ETKDG())
    if ok != 0:
        ok = AllChem.EmbedMolecule(mol3, useRandomCoords=True)
        if ok != 0:
            raise ValueError("self_couple: Embed failed")

    # re-find tags
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

    # small relative shift of second copy to avoid immediate overlap
    second_copy_atoms = list(range(offset, mol3.GetNumAtoms()))
    conf = mol3.GetConformer()
    import numpy as np
    p1 = np.array(conf.GetAtomPosition(new_c1))
    p2 = np.array(conf.GetAtomPosition(new_c2))
    v = p2 - p1
    nv = np.linalg.norm(v)
    if nv < 1e-6:
        v = np.array([1.0, 0.0, 0.0])
        nv = 1.0
    v = v / nv
    shift = 1.42 * v
    for i in second_copy_atoms:
        pi = np.array(conf.GetAtomPosition(i))
        conf.SetAtomPosition(i, pi + shift)

    # add C–C bond if absent
    if mol3.GetBondBetweenAtoms(new_c1, new_c2) is None:
        rw2 = Chem.RWMol(mol3)
        rw2.AddBond(new_c1, new_c2, Chem.BondType.SINGLE)
        mol4 = rw2.GetMol()
    else:
        mol4 = mol3

    mol_tmp = Chem.RemoveHs(mol4)
    # a second embedding can help relax geometry before UFF
    AllChem.EmbedMolecule(mol_tmp, AllChem.ETKDG())
    mol_tmp = Chem.AddHs(mol_tmp)
    AllChem.UFFOptimizeMolecule(mol_tmp)
    return Chem.RemoveHs(mol_tmp)

# -------------------------- worker entry --------------------------
def worker_run(task, precs):
    """
    kind: "self" | "diff"
    """
    try:
        if task["kind"] == "self":
            i = task["i"]
            mode = task["mode"]
            mb, s1, h1 = precs[i]
            mol = _self_couple_from_block(mb, mode)
            new_h = mol_hash16(mol)
            molblock = Chem.MolToMolBlock(mol)
            return ("OK", {"new_hash": new_h, "molblock": molblock,
                           "mode": ("self_sym" if mode in ("C2", "mirror") else "self_as"),
                           "s1": s1, "h1": h1, "s2": s1, "h2": h1})

        else:  # "diff"
            i, j = task["i"], task["j"]
            mb1, s1, h1 = precs[i]
            mb2, s2, h2 = precs[j]
            mol = _couple_simple_from_blocks(mb1, mb2)
            new_h = mol_hash16(mol)
            molblock = Chem.MolToMolBlock(mol)
            return ("OK", {"new_hash": new_h, "molblock": molblock,
                           "mode": "diff", "s1": s1, "h1": h1, "s2": s2, "h2": h2})
    except Exception as e:
        return ("ERR", str(e))

# -------------------------- main runner --------------------------
def auto_run(in_dir: str, target_count: int, root_out: str = ROOT_OUT, n_workers: int | None = None):
    print("=== Coupling started ===")

    # ----- scan precursors -----
    precs = []  # list of (molblock, sym_dir, hash16)
    for sub in os.listdir(in_dir):
        d = os.path.join(in_dir, sub)
        if not os.path.isdir(d):
            continue
        files = [f for f in os.listdir(d) if f.lower().endswith(".mol")]
        for fn in files:
            path = os.path.join(d, fn)
            try:
                mb = load_molblock(path)
                mol = Chem.MolFromMolBlock(mb, sanitize=False)
                if mol is None:
                    continue
                h = mol_hash16(mol)
                precs.append((mb, sub, h))
            except Exception:
                pass

    print("precursors =", len(precs))
    if len(precs) < 2:
        print("Too few precursors.")
        return

    # ----- output dirs, CSV -----
    self_root   = os.path.join(root_out, "self_coupling")
    self_sym_d  = os.path.join(self_root, "symmetric")
    self_as_d   = os.path.join(self_root, "asymmetric")
    diff_d      = os.path.join(root_out, "different_coupling")
    for d in (self_sym_d, self_as_d, diff_d):
        os.makedirs(d, exist_ok=True)

    dbfile = os.path.join(root_out, "coupling_once.csv")
    if not os.path.exists(dbfile):
        with open(dbfile, "w", newline="") as f:
            csv.writer(f).writerow(["coupling_mode", "sym1", "hash1", "sym2", "hash2", "new_hash"])

    # ----- load existing -----
    existing = set()
    for subdir in (diff_d, self_sym_d, self_as_d):
        for fn in os.listdir(subdir):
            if fn.endswith(".mol") and len(fn.split(".")[0]) >= 16:
                existing.add(fn.split(".")[0])

    print("existing =", len(existing))
    cnt = len(existing)

    # ----- workers -----
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    print("workers:", n_workers)

    inflight_cap = MAX_INFLIGHT or (2 * n_workers)

    # ----- task generator -----
    def make_task():
        """Randomly create a task description dict."""
        if random.random() < 0.5:
            # self
            i = random.randrange(0, len(precs))
            mode = random.choice(["C2", "mirror", "asymmetric"])
            return {"kind": "self", "i": i, "mode": mode}
        else:
            # diff
            i, j = random.sample(range(len(precs)), 2)
            return {"kind": "diff", "i": i, "j": j}

    # ----- executor loop with bounded in-flight, per-task timeout -----
    pending = set()
    produced = 0

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as ex:
        # prime some tasks
        while len(pending) < inflight_cap:
            t = make_task()
            fut = ex.submit(worker_run, t, precs)
            pending.add(fut)

        while cnt < target_count:
            if not pending:
                # should not happen, but keep safe
                t = make_task()
                pending.add(ex.submit(worker_run, t, precs))
                continue

            done, pending = wait_first(pending, timeout=TIME_LIMIT_SEC)
            if not done:
                # no task finished within TIME_LIMIT_SEC -> cancel one & requeue
                to_cancel = next(iter(pending))
                to_cancel.cancel()
                pending.remove(to_cancel)
                t = make_task()
                pending.add(ex.submit(worker_run, t, precs))
                print("[TIMEOUT] one task cancelled & requeued")
                continue

            for fut in done:
                try:
                    ok, payload = fut.result(timeout=0.1)
                except TimeoutError:
                    ok, payload = ("ERR", "TimeoutError on result()")
                except Exception as e:
                    ok, payload = ("ERR", f"future exception: {e}")

                # keep pipeline full
                if len(pending) < inflight_cap:
                    pending.add(ex.submit(worker_run, make_task(), precs))

                if ok != "OK":
                    print("[SKIP]", payload)
                    continue

                new_h   = payload["new_hash"]
                molblk  = payload["molblock"]
                mode    = payload["mode"]
                s1, h1  = payload["s1"], payload["h1"]
                s2, h2  = payload["s2"], payload["h2"]

                if new_h in existing:
                    print("[DUP]", new_h)
                    continue

                # write files (rebuild Mol from MolBlock in main process)
                try:
                    m = Chem.MolFromMolBlock(molblk, sanitize=False)
                    if m is None:
                        print("[SKIP] MolFromMolBlock returned None")
                        continue

                    if mode == "diff":
                        out_sub = diff_d
                    else:
                        out_sub = self_sym_d if mode != "self_as" else self_as_d

                    prefix = os.path.join(out_sub, new_h)
                    Chem.MolToMolFile(m, prefix + ".mol")

                    # XYZ block may fail for exotic valence; protect it
                    try:
                        xyz_block = Chem.MolToXYZBlock(m)
                        with open(prefix + ".xyz", "w", encoding="utf-8") as f:
                            f.write(xyz_block)
                    except Exception:
                        pass

                    with open(dbfile, "a", newline="") as f:
                        csv.writer(f).writerow([mode, s1, h1, s2, h2, new_h])

                    existing.add(new_h)
                    cnt += 1
                    produced += 1
                    print(f"[OK] {new_h}  ({cnt}/{target_count}) mode={mode}")

                except Exception as e:
                    print("[SKIP] write failed:", e)
                    continue

    print("DONE")

def wait_first(pending_futures, timeout: float):
    """Wait until at least one future completes or timeout reached."""
    # emulate "as_completed" with timeout that ensures progress
    done = set()
    not_done = set()
    try:
        for fut in as_completed(pending_futures, timeout=timeout):
            done.add(fut)
            # drain whatever else finished immediately
            try:
                while True:
                    fut2 = next(as_completed(pending_futures - done, timeout=0))
                    done.add(fut2)
            except StopIteration:
                pass
            except TimeoutError:
                pass
            break  # at least one done -> break outer
    except TimeoutError:
        # none finished within timeout
        not_done = set(pending_futures)
        return done, not_done

    not_done = set(pending_futures) - done
    return done, not_done

if __name__ == "__main__":
    mp.freeze_support()
    auto_run(IN_DIR, TARGET_COUNT, ROOT_OUT)
    if PAUSE_AT_END:
        try:
            input("\nPress Enter to exit...")
        except EOFError:
            pass
