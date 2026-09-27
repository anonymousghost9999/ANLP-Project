# 1. Transfer code to Ada (from your local machine with IIIT VPN active):
rsync -avz --exclude='*.pt' --exclude='*.joblib' ./ phanindra.gollapalli@ada.iiit.ac.in:~/prompt_scorer_hpc/

# 2. SSH into Ada:
ssh -X phanindra.gollapalli@ada.iiit.ac.in
cd ~/prompt_scorer_hpc

# 3. Setup environment (one-time):
bash setup_ada_env.sh

# 4. Submit SLURM batch job:
sbatch job.slurm

# 5. Monitor progress:
squeue -u $USER
tail -f logs_*.out
