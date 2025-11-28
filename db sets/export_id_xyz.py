from ase.db import connect
from ase.io import write

# change here
start_id = 10
end_id   = 30   # including 30 and 10

db = connect("mol_0D.db")

for i in range(start_id, end_id + 1):
    atoms = db.get(id=i).toatoms()
    write(f"mol_{i}.xyz", atoms)