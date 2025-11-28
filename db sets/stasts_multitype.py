from ase.db import connect
import matplotlib.pyplot as plt
from collections import Counter

# ================= User configuration =================
DB_PATH = "mol_0D.db"

# Switches: enable or disable each statistics module
DO_A = True   # Symmetry distribution + average ring count
DO_B = True   # Doping (N/B/O/S) statistics
DO_C = True   # Total rings vs number of atoms
DO_D = True   # Ring composition (e.g. hexagon fraction)
DO_E = True   # Defect type statistics
# =====================================================

db = connect(DB_PATH)

def get_total_rings(row):
    """
    Compute the total number of rings for a molecule.
    Assumes the database contains n4, n5, n6, n7, n8 columns.
    """
    return row.n4 + row.n5 + row.n6 + row.n7 + row.n8

# ================= A. Symmetry Statistics =================
def stats_symmetry_and_rings():
    """
    1) Count number of molecules for each symmetry class.
    2) Compute average total ring number for each symmetry.
    """
    sym_counts = Counter()
    sym_rings_sum = Counter()

    for row in db.select():
        sym = getattr(row, "sym", "unknown")
        total_rings = get_total_rings(row)

        sym_counts[sym] += 1
        sym_rings_sum[sym] += total_rings

    syms = sorted(sym_counts.keys())
    counts = [sym_counts[s] for s in syms]
    avg_rings = [sym_rings_sum[s] / sym_counts[s] for s in syms]

    # Plot: symmetry count
    plt.figure()
    plt.bar(syms, counts)
    plt.xlabel("Symmetry")
    plt.ylabel("Count")
    plt.title("Symmetry Distribution")

    # Plot: average ring count per symmetry
    plt.figure()
    plt.bar(syms, avg_rings)
    plt.xlabel("Symmetry")
    plt.ylabel("Average Total Rings")
    plt.title("Average Ring Count for Each Symmetry")

    print("=== A. Symmetry statistics ===")
    for s in syms:
        print(f"{s:12s} -> Count: {sym_counts[s]:6d}, "
              f"Avg rings: {sym_rings_sum[s]/sym_counts[s]:.2f}")

# ================= B. Doping Statistics =================
def stats_doping():
    """
    Analyze doping types based on nN, nB, nO, nS columns.
    Separate: none, single doping, and co-doping types.
    """
    doping_counts = Counter()

    for row in db.select():
        flags = []

        if getattr(row, "nN", 0) > 0:
            flags.append("N")
        if getattr(row, "nB", 0) > 0:
            flags.append("B")
        if getattr(row, "nO", 0) > 0:
            flags.append("O")
        if getattr(row, "nS", 0) > 0:
            flags.append("S")

        if not flags:
            label = "none"
        else:
            label = "+".join(sorted(flags))

        doping_counts[label] += 1

    labels = sorted(doping_counts.keys(), key=lambda x: (x != "none", x))
    counts = [doping_counts[l] for l in labels]

    plt.figure()
    plt.bar(labels, counts)
    plt.xlabel("Doping Type")
    plt.ylabel("Count")
    plt.title("Doping Distribution (N/B/O/S)")
    plt.xticks(rotation=45)

    print("=== B. Doping statistics ===")
    for l in labels:
        print(f"{l:10s} -> Count: {doping_counts[l]:6d}")

# ================= C. Total Rings vs Atom Count =================
def stats_rings_vs_natoms():
    """
    Scatter plot of total ring count vs number of atoms.
    """
    total_rings_list = []
    natoms_list = []

    for row in db.select():
        total_rings = get_total_rings(row)
        natoms = getattr(row, "natoms", None)

        if natoms is None:
            continue

        total_rings_list.append(total_rings)
        natoms_list.append(natoms)

    plt.figure()
    plt.scatter(total_rings_list, natoms_list, s=8)
    plt.xlabel("Total Rings")
    plt.ylabel("Number of Atoms")
    plt.title("Total Rings vs Number of Atoms")

    if total_rings_list:
        avg_r = sum(total_rings_list) / len(total_rings_list)
        avg_n = sum(natoms_list) / len(natoms_list)
        print("=== C. Rings vs Atom count ===")
        print(f"Number of samples: {len(total_rings_list)}")
        print(f"Average total rings: {avg_r:.2f}")
        print(f"Average atom count: {avg_n:.2f}")

# ================= D. Ring Composition Statistics =================
def stats_ring_composition():
    """
    Analyze ring composition.
    Example: fraction of hexagons = n6 / total_rings.
    """
    fractions = []

    for row in db.select():
        total = get_total_rings(row)
        if total == 0:
            continue

        frac_hex = row.n6 / total
        fractions.append(frac_hex)

    if not fractions:
        print("No data for ring composition.")
        return

    plt.figure()
    plt.hist(fractions, bins=20)
    plt.xlabel("Hexagon Fraction (n6 / total_rings)")
    plt.ylabel("Count")
    plt.title("Distribution of Hexagon Fraction")

    print("=== D. Ring composition ===")
    print(f"Samples: {len(fractions)}")
    print(f"Average hexagon fraction: {sum(fractions)/len(fractions):.3f}")

# ================= E. Defect Type Statistics =================
def stats_defects():
    """
    Count different defect types from the 'defects' column.
    """
    defect_counts = Counter()

    for row in db.select():
        defect = getattr(row, "defects", "none")
        if defect in ("", None):
            defect = "none"
        defect_counts[defect] += 1

    labels = sorted(defect_counts.keys())
    counts = [defect_counts[l] for l in labels]

    plt.figure()
    plt.bar(labels, counts)
    plt.xlabel("Defect Type")
    plt.ylabel("Count")
    plt.title("Defect Type Distribution")
    plt.xticks(rotation=45)

    print("=== E. Defect statistics ===")
    for d in labels:
        print(f"{d:12s} -> Count: {defect_counts[d]:6d}")

# ================= Main execution =================
if __name__ == "__main__":
    print(f"Database: {DB_PATH}")
    print(f"Total entries: {len(db)}")

    if DO_A:
        stats_symmetry_and_rings()
    if DO_B:
        stats_doping()
    if DO_C:
        stats_rings_vs_natoms()
    if DO_D:
        stats_ring_composition()
    if DO_E:
        stats_defects()

    plt.show()
