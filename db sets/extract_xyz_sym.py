import os
import random
from ase.db import connect
from ase.io import write
from collections import defaultdict

DB_PATH = "mol_0D.db"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "sel_xyz_files")
SAMPLES_PER_SYM = 30
MAX_TOTAL_RINGS = 60 
# RANDOM_SEED = 

# random.seed()

os.makedirs(OUTPUT_DIR, exist_ok=True)
db = connect(DB_PATH)

def total_rings(row):
    return row.n4 + row.n5 + row.n6 + row.n7 + row.n8
# Group rows by symmetry (only <= 60 rings)
sym_groups = defaultdict(list)

for row in db.select():
    tr = total_rings(row)
    # Skip if ring number exceeds the limit
    if tr > MAX_TOTAL_RINGS:
        continue

    sym = getattr(row, "sym", "unknown")
    sym_groups[sym].append(row.id)

print(f"Found {len(sym_groups)} symmetry groups under Ring limit: total_rings <= {MAX_TOTAL_RINGS}.")

for sym, ids in sym_groups.items():
    print(f"{sym}: {len(ids)} molecules (<= {MAX_TOTAL_RINGS} rings)")
print("\nSampling molecules and exporting...")

for sym, ids in sym_groups.items():
    print(f"\nProcessing symmetry: {sym}")
    if len(ids) == 0:
        print("No molecules satisfy the constraint, skipping.")
        continue
    if len(ids) <= SAMPLES_PER_SYM:
        sampled_ids = ids
        print(f"Only {len(ids)} available, exporting all.")
    else:
        sampled_ids = random.sample(ids, SAMPLES_PER_SYM)

    for mol_id in sampled_ids:
        row = db.get(id=mol_id)
        atoms = row.toatoms()
        filename = f"{sym}_{mol_id}.xyz"
        filepath = os.path.join(OUTPUT_DIR, filename)
        write(filepath, atoms)

print(f"Output directory: {OUTPUT_DIR}")