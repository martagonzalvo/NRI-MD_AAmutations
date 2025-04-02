from pathlib import Path
import shutil
import MDAnalysis as mda

# Create a set of directories for all possible residue indices in the parent protein
# This is necessary for the repacking step in Rosetta
# This script is run in the directory where the parent protein is located
# The parent protein is assumed to be named "parent.pdb"

# Get the parent protein and its sequence
parent = mda.Universe("parent.pdb")
parent_seq = parent.select_atoms("protein").residues.resnames

print("Parent protein sequence: ", parent_seq)
print(len(parent_seq))

Path("./repacking-dirs").mkdir(parents=True, exist_ok=True)

RESFILE_TEMPLATE = """NATAA

start
{position} A PIKAA A
"""


for i, parent_resname in enumerate(parent_seq):
    # Check if the residue is named ALA, and skip if so
    if parent_resname == "ALA":
        continue
    # Create a directory for each residue index
    dirname = "./repacking-dirs/" + parent_resname + str(i+1)
    Path(dirname).mkdir(parents=True, exist_ok=True)
    # Copy the parent protein to the directory
    shutil.copy("parent.pdb", dirname)
    # Create a resfile for the residue index
    resfile = RESFILE_TEMPLATE.format(position=i+1)
    Path(dirname + "/resfile").write_text(resfile)




