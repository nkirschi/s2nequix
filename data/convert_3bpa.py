from ase.io import read
from ase.db import connect

import numpy as np
import os


for split_name in [
    "train_300K",
    "train_mixedT",
    "test_300K",
    "test_600K",
    "test_1200K",
    "test_dih",
]:
    xyz_file = f"data/{split_name}.xyz"
    aselmdb_dir = f"data/3bpa/{split_name}"
    os.makedirs(aselmdb_dir, exist_ok=True)
    atoms_list = read(xyz_file, index=":")
    with connect(f"{aselmdb_dir}/data.aselmdb", type="aselmdb") as db:
        for atoms in atoms_list:
            db.write(atoms, data=atoms.info)
    np.savez(
        f"{aselmdb_dir}/metadata.npz",
        natoms=np.repeat(27, len(atoms_list)),
        data_ids=np.repeat("3bpa", len(atoms_list)),
    )
    os.remove(xyz_file)
    print(f"Converted {xyz_file} and saved to {aselmdb_dir}/data.aselmdb")
