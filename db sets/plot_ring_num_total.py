from ase.db import connect
import matplotlib.pyplot as plt
import collections

db = connect("mol_0D.db")

bin_size = 6
ring_bins = collections.Counter()

total_rings_all = []
for row in db.select():
    total = row.n4 + row.n5 + row.n6 + row.n7 + row.n8
    total_rings_all.append(total)

    bin_left = (total // bin_size) * bin_size
    ring_bins[bin_left] += 1

avg_rings = sum(total_rings_all) / len(total_rings_all)

print("Total molecules:", len(total_rings_all))
print("Average total rings:", avg_rings)

sorted_bins = sorted(ring_bins.items())
x = [k for k, v in sorted_bins]
y = [v for k, v in sorted_bins]

plt.figure(figsize=(8,5))
plt.bar(x, y, width=5)
plt.xticks(x, [f"{i}-{i+5}" for i in x], rotation=45)
plt.xlabel("Total rings (bin size = 6)")
plt.ylabel("Count")
plt.title(f"Ring Distribution (avg = {avg_rings:.2f})")
plt.tight_layout()
plt.show()