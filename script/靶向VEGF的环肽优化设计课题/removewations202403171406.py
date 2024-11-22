import os
from pymol import cmd

cmd.load("complex.pdb")
cmd.remove("sol")
cmd.remove("inorganic")
cmd.remove("resn CLA")
cmd.save("complex1.pdb", "complex")

os.remove("complex.pdb")
os.rename("complex1.pdb", "complex.pdb")