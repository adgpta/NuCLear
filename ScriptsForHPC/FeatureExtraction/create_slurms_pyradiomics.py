import os
import glob
import shutil

#print [name for name in os.listdir(".") if os.path.isdir(name)]
raw_dir = "/mnt/sds-hd/sd16j005/Aamrita_Jennifer/LiviasData/Raw/Raw"
mask_dir = "/mnt/sds-hd/sd16j005/Aamrita_Jennifer/LiviasData/Raw/Stardist"

# Change rootdir as you want:
counter = 0

#List all files in raw_dir
for raw_file in os.listdir(raw_dir):
    raw_filename = os.path.join(raw_dir, raw_file)
    mask_filename = os.path.join(mask_dir,raw_file)
    f = open("{}.slurm".format(raw_file), "w")
    f.write("#!/bin/sh\n")
    f.write("########## Begin SLURM header ##########\n")
    f.write("#SBATCH --job-name=\"{}\"\n".format(raw_file))
    f.write("#\n")
    f.write("# Request number of nodes and CPU cores per node for job\n")
    #f.write("#SBATCH --partition=gpu-single\n")
    #f.write("#SBATCH --gres=gpu:1\n")
    #f.write("#SBATCH --cpus-per-gpu=32\n")
    #f.write("#SBATCH --ntasks-per-node=32\n")
    #f.write("#SBATCH --time=8:00:00\n")
    #f.write("#SBATCH --mem=128gb\n")
    f.write("#SBATCH --partition=single\n")
    #f.write("#SBATCH --exclusive\n")
    f.write("#SBATCH --ntasks-per-node=16\n")
    f.write("#SBATCH --time=6:00:00\n")
    f.write("#SBATCH --mem=128gb\n")
    #f.write("#SBATCH --reservation=smart_power\n")
    f.write("#SBATCH -o ./slurm_%j_Amrita_test_cluster_gpu.log\n")
    f.write("#SBATCH -e ./slurm_%j_Amrita_test_cluster_gpu.err\n")
    f.write("#SBATCH --mail-type=ALL\n")
    f.write("#SBATCH --mail-user=johannes.knabbe@uni-heidelberg.de\n")
    f.write("########### End SLURM header ##########\n")
    f.write("export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1\n")
    f.write("export OMP_NUM_THREADS=16\n")
    f.write("eval \"$($HOME/miniconda/bin/conda shell.bash hook)\"\n")
    f.write("conda activate process_pyrad\n")
    f.write("touch test\n")
    f.write("python extract_features-conn-comp_v4-server.py -i {} -m {} -o ./{} --threads 6 > ./{}.out 2>&1\n".format(raw_filename, mask_filename, raw_file, raw_file))
    f.write("exit")
    f.close()
    counter += 1
f.close()


if counter > 0:
        f = open("submitall.sh","w")
        f.write("#!/bin/bash\n")
        for file in glob.glob("*.slurm"):
                f.write("sbatch {}\n".format(file))
        f.close()
