#!/bin/bash
#SBATCH --job-name=cgr
#SBATCH -p gpu --gres=gpu:1
#SBATCH --exclude hendrixgpu17fl,hendrixgpu18fl,hendrixgpu07fl,hendrixgpu08fl,hendrixgpu05fl,hendrixgpu06fl
#SBATCH --time=23:30:00
#SBATCH --ntasks=1 --cpus-per-task=2 --mem=20GB
echo $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES

. /etc/profile.d/modules.sh
eval "$(conda shell.bash hook)"
conda activate /home/jnf811/.conda/envs/yongcao

# ind tes debug
DATASET=tes
if [ "$DATASET" == "ind" ]; then
  CONFIG=config/CGR_in.yaml
elif [ "$DATASET" == "tes" ]; then
  CONFIG=config/CGR.yaml
elif [ "$DATASET" == "debug" ]; then
  CONFIG=config/debug.yaml
fi
# bert-base-chinese / swtx/ernie-3.0-base-chinese /
TEXT_ENCODER=bert-base-chinese
# 42 19960127
SEED=42
EXPER_MODE=baseline_cls
OUTPUT_DIR="output"
TRAIN_BATCH_SIZE=16
TEST_BATCH_SIZE=16
OUTPUT_DIR_PATH="$OUTPUT_DIR"/"$DATASET"_"$EXPER_MODE"_seed_"$SEED"_bs_"$TRAIN_BATCH_SIZE"
EAELY_STOP_THRESH=3
GPU_NUM=1
#SBATCH --output=logs/${DATASET}_${EXPER_MODE}_${TRAIN_BATCH_SIZE}.out

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=$GPU_NUM \
  --master_port $(expr $RANDOM + 1000) train_eval.py \
  --config $CONFIG \
  --text_encoder $TEXT_ENCODER \
  --output_dir $OUTPUT_DIR_PATH \
  --exper_mode $EXPER_MODE \
  --batch_size_train $TRAIN_BATCH_SIZE \
  --batch_size_test $TEST_BATCH_SIZE \
  --early_stop_thresh $EAELY_STOP_THRESH \
  --seed $SEED