from ase import units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from ase.db import connect

import fileinput
import numpy as np
import os


KCAL_TO_EV = units.kcal / units.mol

for split_name in [
    "buckyball-catcher",
    "double-walled_nanotube",
]:
    xyz_file = f"data/md22_{split_name}.xyz"
    aselmdb_dir = f"data/md22/{split_name}"
    with fileinput.FileInput(xyz_file, inplace=True) as file:
        for line in file:
            # Target the comment lines containing the properties
            if "Energy=" in line and "Properties=" in line:
                print(line.replace("Energy=", "energy="), end="")
            else:
                print(line, end="")
    os.makedirs(aselmdb_dir, exist_ok=True)
    atoms_list = read(xyz_file, index=":")
    with connect(f"{aselmdb_dir}/data.aselmdb", type="aselmdb") as db:
        for atoms in atoms_list:
            converted_energy = atoms.get_potential_energy() * KCAL_TO_EV
            converted_forces = atoms.get_forces() * KCAL_TO_EV
            atoms.calc = SinglePointCalculator(
                atoms, energy=converted_energy, forces=converted_forces
            )
            db.write(atoms)
    np.savez(
        f"{aselmdb_dir}/metadata.npz",
        natoms=np.repeat(len(atoms_list[0]), len(atoms_list)),
        data_ids=np.repeat("md22", len(atoms_list)),
    )
    os.remove(xyz_file)
    print(f"Converted {xyz_file} and saved to {aselmdb_dir}/data.aselmdb")
