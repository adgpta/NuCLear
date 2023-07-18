import os
import glob
#print [name for name in os.listdir(".") if os.path.isdir(name)]
inputdir = "/mnt/sds-hd/sd16j005/JohannesJorisLivia/ElifeRevision/Computing/23-08-2022"
print(inputdir)
counter = 0
for name in os.listdir(inputdir):
        print(name)
        if os.path.isdir(os.path.join(inputdir,name)) and not ("Output" in name):
                curr_dir = os.path.join(inputdir, name)
                print(curr_dir)
                output_dir = "{}Output".format(curr_dir)
                f = open("{}.slurm".format(name), "w")
                f.write("#!/bin/sh\n")
                f.write("########## Begin SLURM header ##########\n")
                f.write("#SBATCH --job-name=\"{}\"\n".format(name))
                f.write("#\n")
                f.write("# Request number of nodes and CPU cores per node for job\n")
                f.write("#SBATCH --partition=single\n")
                #f.write("#SBATCH --constraint=sky\n")
                f.write("#SBATCH --gres=gpu:1\n")
                #f.write("#SBATCH --cpus-per-gpu=8\n")
                #f.write("#SBATCH --gres=gpu:gpu-cas:1\n")
                f.write("#SBATCH --ntasks-per-node=32\n")
                f.write("#SBATCH --time=12:00:00\n")
                f.write("#SBATCH --mem=125gb\n")
                f.write("#SBATCH -o ./slurm_%j_Amrita_test_cluster_gpu.log\n")
                f.write("#SBATCH -e ./slurm_%j_Amrita_test_cluster_gpu.err\n")
                f.write("#SBATCH --mail-type=ALL\n")
                f.write("#SBATCH --mail-user=johannes.knabbe@uni-heidelberg.de\n")
                f.write("########### End SLURM header ##########\n")
                f.write("module load devel/cuda/11.6\n")
                f.write("module load lib/cudnn/8.5.0-cuda-11.6\n")
                f.write("module load devel/miniconda/3\n")
                #f.write("cd $HOME/stardist/\n")
                f.write("export USER=Amrita\n")
                #f.write("eval \"$($HOME/miniconda/bin/conda shell.bash hook)\"\n")
                f.write("conda activate csbdeep\n")
                f.write("export OMP_NUM_THREADS=32\n")
                f.write("export TF_FORCE_GPU_ALLOW_GROWTH=true\n")
                #f.write("cd $HOME\n")
                f.write("predict_stardist_3d -i {} -n  /mnt/sds-hd/sd16j005/Aamrita_Jennifer/Johannes/Model/15_May_2021 -m stardist -o {} -r 80 --ext tif > ./{}.out 2>&1\n".format(curr_dir, output_dir, name))
                f.write("exit")
                f.close()
        counter += 1

if counter > 0:
        f = open("submitall.sh","w")
        f.write("#!/bin/bash\n")
        for file in glob.glob("*.slurm"):
                f.write("sbatch {}\n".format(file))
        f.close()
