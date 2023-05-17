##!/bin/bash
##SBATCH --job-name=cgr
##SBATCH -p gpu --gres=gpu:1
##SBATCH --exclude hendrixgpu17fl,hendrixgpu18fl,hendrixgpu07fl,hendrixgpu08fl,hendrixgpu05fl,hendrixgpu06fl
##SBATCH --time=23:30:00
##SBATCH --ntasks=1 --cpus-per-task=2 --mem=20GB
#echo $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES
#
#. /etc/profile.d/modules.sh
#eval "$(conda shell.bash hook)"
#conda activate /home/jnf811/.conda/envs/yongcao

CHECKPOINT=output/debug_baseline_cls_seed_42_bs_16/checkpoint_best.pth
DATASET=data/industry_small/test.jsonl
# bert-base-chinese / swtx/ernie-3.0-base-chinese /
TEXT_ENCODER=swtx/ernie-3.0-base-chinese
SEED=42 # 42 19960127
BATCH_SIZE=32

python evaluation.py \
  --checkpoint $CHECKPOINT \
  --test_data $DATASET \
  --text_encoder $TEXT_ENCODER  \
  --seed $SEED \
  --batch_size $BATCH_SIZE