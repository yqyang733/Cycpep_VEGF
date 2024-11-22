from pymol import cmd

cmd.load("system.pdb")
cmd.load_traj("wrapped.dcd",interval=10)
cmd.remove("sol")
cmd.remove("resn SOD")
cmd.remove("resn CLA")

for i in range(1,cmd.count_states("system")+1):
    cmd.save("./ensemble/system_" + str(i) + ".pdb", state = i)