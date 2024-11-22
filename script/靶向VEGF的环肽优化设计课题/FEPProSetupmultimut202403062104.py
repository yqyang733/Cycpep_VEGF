import os
import time
import shutil
from collections import defaultdict

class config:

    def __init__(self, input):

        config_dict = dict()
        with open(input) as f:
            for i in f:
                config_dict[i.split(":")[0].strip()] = i.split(":")[1].replace("\n", "").strip()

        self.bond_pdb = os.path.join(config_dict["bond_pdb"])
        self.bond_psf = os.path.join(config_dict["bond_psf"])
        self.free_pdb = os.path.join(config_dict["free_pdb"])
        self.free_psf = os.path.join(config_dict["free_psf"])
        self.segname = config_dict["segname"]
        #self.resid = config_dict["resid"]
        self.mutation = config_dict["mutation"]
        self.vmd_path = os.path.join(config_dict["vmd_path"])
        self.ff_path = os.path.join(config_dict["ff_path"])
        self.submit = config_dict["submit_mode"]
        self.duplicate = config_dict["duplicate"]

class Lig_pdb:

    def __init__(self, lig_pdb_file):

        self.segname = set() # all segment names
        self.segresid_atoms = defaultdict(list)  # key: (seg, resid)   value: [atoms,]

        with open(lig_pdb_file) as f:
            f1 = f.readlines()
        for i in f1:
            if i.startswith("ATOM"):
                self.segname.add(i[66:76].strip())
                self.segresid_atoms[(i[66:76].strip(), i[6:11].strip())].append(i[12:16].strip())

def do_mutate(pdb, segname, ff_path, mutation, flag, vmd_path):

    resids = "("
    muts = ""
    resid_num = []
    for i in range(len(mutation)-1):
        mut_before = mutation[i][0]
        mut_after = mutation[i][-1]
        resi = mutation[i][1:-1]
        resid_num.append(resi)
        resids = resids + "resid " + resi + " or "
        muts = muts + "  mutate {0} {1}\n".format(resi, mut_before+str(2)+mut_after)
    mut_before = mutation[len(mutation)-1][0]
    mut_after = mutation[len(mutation)-1][-1]
    resi = mutation[len(mutation)-1][1:-1]
    resid_num.append(resi)
    resids = resids + "resid " + resi + ")"
    muts = muts + "  mutate {0} {1}".format(resi, mut_before+str(2)+mut_after)

    mk_mut = open("mk_mut_{}.tcl".format(flag), "w")
    mk_mut.write(
'''
package require psfgen

mol new {0}
set sel [atomselect top "segname {1} and {2} and not name C CA N O HN HA CB"]
foreach name [$sel get name] {{
    set sel [atomselect top "segname {1} and {2} and name $name"]
    $sel set name ${{name}}A
}}
set sel [atomselect top "segname {1}"]
$sel writepdb mutant.pdb

resetpsf
topology {3}/top_all36_propatch.rtf
topology {3}/top_all36_prot.rtf
topology {3}/top_all36_hybrid.inp
topology {3}/toppar_water_ions.str

segment MUT {{
  first none
  last CT2
  pdb mutant.pdb
{4}
}}
patch DISU MUT:1 MUT:11
patch CONH MUT:10 MUT:14
coordpdb mutant.pdb MUT
regenerate angles dihedrals
guesscoord

writepsf mutant.psf
writepdb mutant.pdb

quit
'''.format(pdb, segname, resids, ff_path, muts)
    )
    mk_mut.close()
    cmd = vmd_path + " -dispdev text -e mk_mut_{}.tcl".format(flag)
    os.system(cmd)
    time.sleep(1)

    return resid_num

def do_merge(pdb, psf, segname, flag, vmd_path):

    pdb_seg = Lig_pdb(pdb)
    pdb_seg = [i for i in pdb_seg.segname if i != segname]
    pdb_seg = " ".join(pdb_seg) 
    mk_merge = open("mk_merge_{}.tcl".format(flag), "w")
    mk_merge.write(
'''
package require topotools

mol new mutant.psf
mol addfile mutant.pdb
mol new {0}
mol addfile {1}

set sel1 [atomselect 0 all]
set sel2 [atomselect 1 "segname {2}"]
set mol [::TopoTools::selections2mol "$sel1 $sel2"]
animate write psf complex-fep.psf $mol
animate write pdb complex.pdb $mol
    
quit
'''.format(psf, pdb, pdb_seg)
    )
    mk_merge.close()
    cmd = vmd_path + " -dispdev text -e mk_merge_{}.tcl".format(flag)
    os.system(cmd)
    time.sleep(1)

def mark_fep(resid_nums):

    mark_atoms = ["N", "HN", "CA", "HA", "C", "O"]

    with open("complex.pdb") as f:
        f1 = f.readlines()

    rt = open("complex-fep.pdb", "w")
    for i in f1:
        if i[66:76].strip() == "MUT" and i[22:26].strip() in resid_nums:
            if i[12:16].strip() in mark_atoms:
                rt.write(i)
            elif i[12:16].strip()[-1] == "A":
                rt.write(i[:60] + "{:6.2f}".format(-1) + i[66:])
            elif i[12:16].strip()[-1] == "B":
                rt.write(i[:60] + "{:6.2f}".format(1) + i[66:])
        else:
            rt.write(i)
    rt.close()

def md_pbc_box(vmd_path):
    mk_pbcbox = open("mk_pbcbox.tcl", "w")
    mk_pbcbox.write(
"""
#!/bin/bash
# vmd -dispdev text -e mk_pbcbox.tcl

package require psfgen
psfcontext reset
mol load psf complex-fep.psf pdb complex-fep.pdb
set everyone [atomselect top all]
set minmax [measure minmax $everyone]
foreach {min max} $minmax { break }
foreach {xmin ymin zmin} $min { break }
foreach {xmax ymax zmax} $max { break }

set file [open "PBCBOX.dat" w]
puts $file "cellBasisVector1 [ expr $xmax - $xmin ] 0 0 "
puts $file "cellBasisVector2 0 [ expr $ymax - $ymin ] 0 "
puts $file "cellBasisVector3 0 0 [ expr $zmax - $zmin ] "
puts $file "cellOrigin [ expr ($xmax + $xmin)/2 ] [ expr ($ymax + $ymin)/2 ] [ expr ($zmax + $zmin)/2 ] "

exit
"""
    )
    mk_pbcbox.close()
    cmd = vmd_path + " -dispdev text -e mk_pbcbox.tcl"
    os.system(cmd)
    time.sleep(1)

def position_constraints(vmd_path):
    seg_fep = Lig_pdb("complex-fep.pdb")
    seg_fep_new = []
    for i in seg_fep.segname:
        if not i.startswith("WT") and (not i.startswith("ION")):
            seg_fep_new.append(i)
    if not (len(seg_fep_new) == 1 and seg_fep_new[0] == "MUT"):
        mk_bonded = open("bonded_constraints.tcl", "w")
        mk_bonded.write(
"""
mol new complex-fep.pdb type pdb waitfor all
set all [atomselect top "all"]

$all set beta 0
set sel [atomselect top "(((segname PRO) and backbone) or ((segname MUT) and noh))"]
$sel set beta 1
$all writepdb constraints.pdb

quit
"""
    )
        mk_bonded.close()
        cmd = vmd_path + " -dispdev text -e bonded_constraints.tcl"
        os.system(cmd)
        time.sleep(1)
    else:
        mk_free = open("free_constraints.tcl", "w")
        mk_free.write(
"""
mol new complex-fep.pdb type pdb waitfor all
set all [atomselect top "all"]

$all set beta 0
set sel [atomselect top "((segname MUT) and noh)"]
$sel set beta 1
$all writepdb constraints.pdb

quit
"""
    )
        mk_free.close()
        cmd = vmd_path + " -dispdev text -e free_constraints.tcl"
        os.system(cmd)
        time.sleep(1)

def fep_tcl(fep_tcl):

    mk_feptcl = open(fep_tcl, "w")
    mk_feptcl.write(
"""
##############################################################
# FEP SCRIPT
# Jerome Henin <jhenin@ifr88.cnrs-mrs.fr>
#
# Changes:
# 2010-04-24: added runFEPmin
# 2009-11-17: changed for NAMD 2.7 keywords
# 2008-06-25: added TI routines
# 2007-11-01: fixed runFEP to handle backwards transformations
#             (i.e. dLambda < 0)
##############################################################

##############################################################
# Example NAMD input:
#
# source fep.tcl
#
# alch                  on
# alchFile              system.fep
# alchCol               B
# alchOutFreq           10
# alchOutFile           system.fepout
# alchEquilSteps        500
#
# set nSteps      5000
# set init {0 0.05 0.1}
# set end {0.9 0.95 1.0}
#
# runFEPlist $init $nSteps
# runFEP 0.1 0.9 0.1 $nSteps
# runFEPlist $end $nSteps
##############################################################

##############################################################
# proc runFEPlist { lambdaList nSteps }
#
# Run n FEP windows joining (n + 1) lambda-points
##############################################################

proc runFEPlist { lambdaList nSteps } {
    # Keep track of window number
    global win
    if {![info exists win]} {
      set win 1
    }

    set l1 [lindex $lambdaList 0]
    foreach l2 [lrange $lambdaList 1 end] {
      print [format "Running FEP window %3s: Lambda1 %-6s Lambda2 %-6s \[dLambda %-6s\]"\
        $win $l1 $l2 [expr $l2 - $l1]]
      firsttimestep    0
      alchLambda       $l1
      alchLambda2      $l2
      run              $nSteps

      set l1 $l2
      incr win
    }
}

proc runFEPlist_restart { lambdaList nSteps starting timestep } {
    # Keep track of window number
    global win
    if {![info exists win]} {
      set win $starting
    }

    set l1 [lindex $lambdaList $starting]
    foreach l2 [lrange $lambdaList [expr $starting + 1] end] {
      print [format "Running FEP window %3s: Lambda1 %-6s Lambda2 %-6s \[dLambda %-6s\]"\
        $win $l1 $l2 [expr $l2 - $l1]]
      if { $l1 == [lindex $lambdaList $starting] } {
        set firsttimestep $timestep
        alchEquilSteps    0
      } else {
        set firsttimestep 0
        alchEquilSteps    10000
      }
      firsttimestep    $firsttimestep
      alchLambda       $l1
      alchLambda2      $l2
      run              [expr $nSteps - $firsttimestep]

      set l1 $l2
      incr win
    }
}


##############################################################
# proc runFEP { start stop dLambda nSteps }
#
# FEP windows of width dLambda between values start and stop
##############################################################

proc runFEP { start stop dLambda nSteps } {
    set epsilon 1e-15

    if { ($stop < $start) && ($dLambda > 0) } {
      set dLambda [expr {-$dLambda}]
    }

    if { $start == $stop } {
      set ll [list $start $start]
    } else {
      set ll [list $start]
      set l2 [increment $start $dLambda]

      if { $dLambda > 0} {
        # A small workaround for numerical rounding errors
        while { [expr {$l2 <= ($stop + $epsilon) } ] } {
          lappend ll $l2
          set l2 [increment $l2 $dLambda]
        }
      } else {
        while { [expr {$l2 >= ($stop - $epsilon) } ] } {
          lappend ll $l2
          set l2 [increment $l2 $dLambda]
        }
      }
    }

    runFEPlist $ll $nSteps
}


##############################################################
##############################################################

proc runFEPmin { start stop dLambda nSteps nMinSteps temp} {
    set epsilon 1e-15

    if { ($stop < $start) && ($dLambda > 0) } {
      set dLambda [expr {-$dLambda}]
    }

    if { $start == $stop } {
      set ll [list $start $start]
    } else {
      set ll [list $start]
      set l2 [increment $start $dLambda]

      if { $dLambda > 0} {
        # A small workaround for numerical rounding errors
        while { [expr {$l2 <= ($stop + $epsilon) } ] } {
          lappend ll $l2
          set l2 [increment $l2 $dLambda]
        }
      } else {
        while { [expr {$l2 >= ($stop - $epsilon) } ] } {
          lappend ll $l2
          set l2 [increment $l2 $dLambda]
        }
      }
    }

    if { $nMinSteps > 0 } {
      alchLambda       $start
      alchLambda2      $start
      minimize $nMinSteps
      reinitvels $temp
    }

    runFEPlist $ll $nSteps
}

##############################################################
##############################################################

proc runTIlist { lambdaList nSteps } {
    # Keep track of window number
    global win
    if {![info exists win]} {
            set win 1
    }

    foreach l $lambdaList {
            print [format "Running TI window %3s: Lambda %-6s " $win $l ]
            firsttimestep 0
            alchLambda       $l
            run $nSteps
            incr win
    }
}


##############################################################
##############################################################

proc runTI { start stop dLambda nSteps } {
    set epsilon 1e-15

    if { ($stop < $start) && ($dLambda > 0) } {
      set dLambda [expr {-$dLambda}]
    }

    if { $start == $stop } {
      set ll [list $start $start]
    } else {
      set ll [list $start]
      set l2 [increment $start $dLambda]

      if { $dLambda > 0} {
        # A small workaround for numerical rounding errors
        while { [expr {$l2 <= ($stop + $epsilon) } ] } {
          lappend ll $l2
          set l2 [increment $l2 $dLambda]
        }
      } else {
        while { [expr {$l2 >= ($stop - $epsilon) } ] } {
          lappend ll $l2
          set l2 [increment $l2 $dLambda]
        }
      }
    }

    runTIlist $ll $nSteps
}

##############################################################
# Increment lambda and try to correct truncation errors around
# 0 and 1
##############################################################

proc increment { lambda dLambda } {
    set epsilon 1e-15
    set new [expr { $lambda + $dLambda }]

    if { [expr $new > - $epsilon && $new < $epsilon] } {
      return 0.0
    }
    if { [expr ($new - 1) > - $epsilon && ($new - 1) < $epsilon] } {
      return 1.0
    }
    return $new
}
"""
    )

def submit_divide(do_fep, dup, subname):
    mk_submit = open(do_fep, "w")
    mk_submit.write(
"""#!/bin/bash
#SBATCH -J {0}
#SBATCH -p single
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

echo "Start time: $(date)"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Job directory: $(pwd)"

# Decide the software version
source /public/software/profile.d/apps_namd_3.0alpha9.sh

NAMD="namd3 +p1 +devices 0"

cd equil
base=fep-equil.conf
$NAMD $base > $base.log
cd ../prod
base=fep-prod.conf
$NAMD $base > $base.log
""".format(subname + "_" + dup)
    )
    mk_submit.close()

def submit_all(do_fep, duplicate, subname):
    mk_submit = open(do_fep, "w")
    mk_submit.write(
"""#!/bin/bash
#SBATCH -J {1}
#SBATCH -p single
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

echo "Start time: $(date)"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Job directory: $(pwd)"

# Decide the software version
source /public/software/profile.d/apps_namd_3.0alpha9.sh

NAMD="namd3 +p1 +devices 0"

for i in `seq 1 {0}`
do
    cd dup${{i}}
    cd equil
    base=fep-equil.conf
    $NAMD $base > $base.log
    cd ../prod
    base=fep-prod.conf
    $NAMD $base > $base.log
    cd ../../
done
""".format(str(duplicate), subname)
    )
    mk_submit.close()

def fep_equil_config(equil_config, ff_path):
    mk_equil_config = open(equil_config, "w")
    mk_equil_config.write(
"""
#############################################################
## JOB DESCRIPTION                                         ##
#############################################################

# FEP: Minimization and Equilibration (NPT) of
# protein-ligand complex in a Water Box
# namd3 +p1 fep-com-equil.conf > fep-com-equil.log


#############################################################
## ADJUSTABLE PARAMETERS                                   ##
#############################################################

set  temp           310
set  outputbase     complex
set  outputName     $outputbase-equil
firsttimestep       0
# if you do not want to open this option, assign 0
set INPUTNAME       0                      ;# use the former outputName, for restarting a simulation
set CONSPDB     0
set CONSSCALE   1                      ;# default; initial value if you want to change
set parpath     {0}

#############################################################
## SIMULATION PARAMETERS                                   ##
#############################################################

structure           ../../common/complex-fep.psf
coordinates         ../../common/complex-fep.pdb

# Input
paraTypeCharmm      on
parameters          ${{parpath}}/addition.prm
parameters          ${{parpath}}/par_all36m_prot.prm
parameters          ${{parpath}}/toppar_water_ions_namd.str
mergeCrossterms     yes

# restart or PBC
if {{ $INPUTNAME != 0 }} {{
    # restart
    BinVelocities $INPUTNAME.restart.vel.old
    BinCoordinates $INPUTNAME.restart.coor.old
    ExtendedSystem $INPUTNAME.restart.xsc.old
}} else {{
    # Periodic Boundary Conditions
    temperature $temp
    source ../../common/PBCBOX.dat
}}


## Force-Field Parameters
exclude             scaled1-4;         # non-bonded exclusion policy to use "none,1-2,1-3,1-4,or scaled1-4"
                                       # 1-2: all atoms pairs that are bonded are going to be ignored
                                       # 1-3: 3 consecutively bonded are excluded
                                       # scaled1-4: include all the 1-3, and modified 1-4 interactions
                                       # electrostatic scaled by 1-4scaling factor 1.0
                                       # vdW special 1-4 parameters in charmm parameter file.
1-4scaling          1.0

# CONSTANT-T
langevin                on
langevinTemp            $temp
langevinDamping         10.0

# CONSTANT-P, not in tutorial
useGroupPressure        yes;           # use a hydrogen-group based pseudo-molecular viral to calcualte pressure and
                                        # has less fluctuation, is needed for rigid bonds (rigidBonds/SHAKE)
useFlexibleCell         no;            # yes for anisotropic system like membrane
useConstantRatio        no;            # keeps the ratio of the unit cell in the x-y plane constant A=B
#    useConstatntArea     yes;
langevinPiston          on
langevinPistonTarget    1.01325
langevinPistonPeriod    100;         # 100? 2000?
langevinPistonDecay     50;         # 50?
langevinPistonTemp      $temp
StrainRate              0.0 0.0 0.0

# CUT-OFFS
switching                on
switchdist              10.0
cutoff                  12.0
pairlistdist            13.5

PME                     yes
PMEGridSpacing          1.0
PMETolerance            10e-6
PMEInterpOrder          4

wrapWater               on;                # wrap water to central cell
wrapAll                 on;                # wrap other molecules too
wrapNearest             off;               # use for non-rectangular cells (wrap to the nearest image)

# SPACE PARTITIONING
splitpatch              hydrogen
hgroupcutoff            2.8
stepspercycle           20
margin                  2
longSplitting           C2

# RESPA PROPAGATOR
# timestep                1.0
timestep                2.0
useSettle               on
fullElectFrequency      2
nonbondedFreq           1

# SHAKE
rigidbonds              all
rigidtolerance          0.000001
rigiditerations         400

# COM
ComMotion               no

# vdw
vdwForceSwitching       on

#############################################################
## EXECUTION SCRIPT                                        ##
#############################################################

# Output

outputname              $outputName

# 500steps = every 1ps
# not important?
computeEnergies         50
outputenergies          1000
outputtiming            1000
outputpressure          1000
restartfreq             1000
XSTFreq                 1000
binaryoutput            yes
binaryrestart           yes

# Positional restraints
# Write out a separate pdb file in which the B values for
# the backbone, the non-hydrogen nucleotide atoms, the ion,
# and the water oxygens within 2.5 A of magnesium are set to 2
if {{ $CONSPDB != 0 }} {{
    Constraints          yes
    ConsRef              $CONSPDB.pdb
    ConsKFile            $CONSPDB.pdb
    ConskCol             B
    constraintScaling    $CONSSCALE
}}

source                  ../../common/fep.tcl

alch                    on
alchType                FEP
alchFile                ../../common/complex-fep.pdb
alchCol                 B
alchOutFile             $outputName.fepout
alchOutFreq             50

alchVdwLambdaEnd        1.0
alchElecLambdaStart     0.1
alchVdWShiftCoeff       1.0
alchDecouple            on

alchEquilSteps          10000
set numSteps            50000   ;#250000

set numMinSteps         5000

runFEPmin 0.0 0.0 0.0 $numSteps $numMinSteps $temp
""".format(os.path.join(ff_path))
    )
    mk_equil_config.close()

def fep_prod_config(prod_config, ff_path):
    mk_prod_config = open(prod_config, "w")
    mk_prod_config.write(
"""
#############################################################
## JOB DESCRIPTION                                         ##
#############################################################

# FEP: Forward run of
# protein-ligand complex in a Water Box
# namd3 +p1 +devices 0 fep-prod.conf > fep-prod.log

#############################################################
## ADJUSTABLE PARAMETERS                                   ##
#############################################################

set  temp           310
set  outputbase     complex
set  outputName     $outputbase-prod
set  INPUTNAME       0
# use the former outputName, for restarting a simulation
set parpath     {0}

#############################################################
## SIMULATION PARAMETERS                                   ##
#############################################################

structure           ../../common/complex-fep.psf
coordinates         ../../common/complex-fep.pdb

# Input
paraTypeCharmm      on
parameters          ${{parpath}}/addition.prm
parameters          ${{parpath}}/par_all36m_prot.prm
parameters          ${{parpath}}/toppar_water_ions_namd.str
mergeCrossterms yes

# restart or PBC
if {{ $INPUTNAME != 0 }} {{
    # restart
    BinVelocities $INPUTNAME.restart.vel.old
    BinCoordinates $INPUTNAME.restart.coor.old
    ExtendedSystem $INPUTNAME.restart.xsc.old
}} else {{
    # from equil. use the former outputName
    bincoordinates          ../equil/$outputbase-equil.coor
    binvelocities           ../equil/$outputbase-equil.vel
    extendedSystem          ../equil/$outputbase-equil.xsc
}}


## Force-Field Parameters
exclude             scaled1-4;         # non-bonded exclusion policy to use "none,1-2,1-3,1-4,or scaled1-4"
                                       # 1-2: all atoms pairs that are bonded are going to be ignored
                                       # 1-3: 3 consecutively bonded are excluded
                                       # scaled1-4: include all the 1-3, and modified 1-4 interactions
                                       # electrostatic scaled by 1-4scaling factor 1.0
                                       # vdW special 1-4 parameters in charmm parameter file.
1-4scaling              1.0

# CONSTANT-T
langevin                on
langevinTemp            $temp
langevinDamping         1.0

# CONSTANT-P, not in tutorial
useGroupPressure        yes;           # use a hydrogen-group based pseudo-molecular viral to calcualte pressure and
                                        # has less fluctuation, is needed for rigid bonds (rigidBonds/SHAKE)
useFlexibleCell         no;            # yes for anisotropic system like membrane
useConstantRatio        no;            # keeps the ratio of the unit cell in the x-y plane constant A=B
#    useConstatntArea     yes;
langevinPiston          on
langevinPistonTarget    1.01325
langevinPistonPeriod    100;         # 100? 2000?
langevinPistonDecay     50;         # 50?
langevinPistonTemp      $temp
StrainRate              0.0 0.0 0.0

# CUT-OFFS
switching               on
switchdist              10.0
cutoff                  12.0
pairlistdist            13.5

PME                     yes
PMEGridSpacing          1.0
PMETolerance            10e-6
PMEInterpOrder          4

wrapWater               on;                # wrap water to central cell
wrapAll                 on;                # wrap other molecules too
wrapNearest             off;               # use for non-rectangular cells (wrap to the nearest image)

# SPACE PARTITIONING
splitpatch              hydrogen
hgroupcutoff            2.8
stepspercycle           20
margin                  2
longSplitting           C2

# RESPA PROPAGATOR
# timestep                1.0
timestep                2.0
useSettle               on
fullElectFrequency      2
nonbondedFreq           1

# SHAKE
rigidbonds              all
rigidtolerance          0.000001
rigiditerations         400

# COM
# according to P. Blood use "no" for first NPT run
# then use "yes" for all NPT runs afterward
COMmotion               yes

# vdw
vdwForceSwitching       on

CUDASOAintegrate         on


#############################################################
## EXECUTION SCRIPT                                        ##
#############################################################

# Output

outputname              $outputName

# 500steps = every 1ps
# not important?
computeEnergies         50
outputenergies          10200
outputtiming            10200
outputpressure          10200
restartfreq             10200
XSTFreq                 10200
dcdfreq                 154000  # steps. 10 frames/per window
binaryoutput            yes
binaryrestart           yes

source                  ../../common/fep.tcl

alch                    on
alchType                FEP
alchFile                ../../common/complex-fep.pdb
alchCol                 B
alchOutFile             $outputName.fepout
alchOutFreq             50  # 10

alchVdwLambdaEnd        1.0
alchElecLambdaStart     0.1
alchVdWShiftCoeff       1.0
alchDecouple            on

alchEquilSteps          20000
set numSteps            770000  ;# 1.5ns a window

set all {{0.00 0.00001 0.0001 0.001 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.99 0.999 0.9999 0.99999 1.00}}

runFEPlist $all $numSteps
""".format(os.path.join(ff_path))
    )
    mk_prod_config.close()

def run_fep_individe(pdb, psf, segname, ff_path, mutation, flag, vmd_path, duplicate, submit, subname):

    resid_nums = do_mutate(pdb, segname, ff_path, mutation, flag, vmd_path)
    do_merge(pdb, psf, segname, flag, vmd_path)
    mark_fep(resid_nums)
    md_pbc_box(vmd_path)
    position_constraints(vmd_path)
    if os.path.exists(os.path.join(".", "common")):
        shutil.rmtree(os.path.join(".", "common"))
        os.makedirs(os.path.join(".", "common"))
    else:
        os.makedirs(os.path.join(".", "common"))
    shutil.move("mk_mut_{}.tcl".format(flag), os.path.join(".", "common", "mk_mut_{}.tcl".format(flag)))
    shutil.move("mk_merge_{}.tcl".format(flag), os.path.join(".", "common", "mk_merge_{}.tcl".format(flag)))
    shutil.move("complex.pdb", os.path.join(".", "common", "complex.pdb"))
    shutil.move("complex-fep.pdb", os.path.join(".", "common", "complex-fep.pdb"))
    shutil.move("complex-fep.psf", os.path.join(".", "common", "complex-fep.psf"))
    shutil.move("mk_pbcbox.tcl", os.path.join(".", "common", "mk_pbcbox.tcl"))
    shutil.move("PBCBOX.dat", os.path.join(".", "common", "PBCBOX.dat"))
    shutil.move("mutant.pdb", os.path.join(".", "common", "mutant.pdb"))
    shutil.move("mutant.psf", os.path.join(".", "common", "mutant.psf"))
    try:
        shutil.move("bonded_constraints.tcl", os.path.join(".", "common", "bonded_constraints.tcl"))
    except:
        shutil.move("free_constraints.tcl", os.path.join(".", "common", "free_constraints.tcl"))
    shutil.move("constraints.pdb", os.path.join(".", "common", "constraints.pdb"))
    fep_tcl(os.path.join(".", "common", "fep.tcl"))
    for i in range(1, int(duplicate)+1):
        if os.path.exists(os.path.join(".", "dup"+str(i))):
            shutil.rmtree(os.path.join(".", "dup"+str(i)))
            os.makedirs(os.path.join(".", "dup"+str(i), "equil"))
            os.makedirs(os.path.join(".", "dup"+str(i), "prod"))
        else:
            os.makedirs(os.path.join(".", "dup"+str(i), "equil"))
            os.makedirs(os.path.join(".", "dup"+str(i), "prod"))
        fep_equil_config(os.path.join(".", "dup"+str(i), "equil", "fep-equil.conf"), ff_path)
        fep_prod_config(os.path.join(".", "dup"+str(i), "prod", "fep-prod.conf"), ff_path)
        if int(submit) == 0 or (int(submit) == 2):
            submit_divide(os.path.join(".", "dup"+str(i), "do_fep.sh"), "dup"+str(i), subname)
    if int(submit) == 1 or (int(submit) == 2):
        submit_all(os.path.join(".", "do_fep.sh"), duplicate, subname)

def run_fep():
    import sys

    settings = config("settings_multi_mut.dat")
    #settings.resid = sys.argv[1]
    settings.mutation = sys.argv[1]
    settings.mutation = settings.mutation.split(",")
    file_name = "".join(settings.mutation)
    if os.path.exists(os.path.join(".", file_name)):
        shutil.rmtree(os.path.join(".", file_name))
        os.makedirs(os.path.join(".", file_name, "bonded"))
    else:
        os.makedirs(os.path.join(".", file_name, "bonded"))
    os.chdir(os.path.join(".", file_name, "bonded"))
    run_fep_individe(settings.bond_pdb, settings.bond_psf, settings.segname, settings.ff_path, settings.mutation, "bonded", settings.vmd_path, settings.duplicate, settings.submit, file_name+"_b")
    os.chdir(os.path.join(".."))

    os.makedirs(os.path.join(".", "free"))
    os.chdir(os.path.join(".", "free"))
    run_fep_individe(settings.free_pdb, settings.free_psf, settings.segname, settings.ff_path, settings.mutation, "free", settings.vmd_path, settings.duplicate, settings.submit, file_name+"_f")
    os.chdir(os.path.join(".."))

def main():
    run_fep()

if __name__=="__main__":
    main() 
