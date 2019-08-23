CONFIG="configs/gunshot_val.json" 
MODEL_DIR="./models/baseline/"
GPU=7

if [ $# -ne 1 ]; then
    echo Bad arguments
    exit
fi

mkdir -p $(dirname $1)

LATEST_MODEL=$MODEL_DIR$(ls -1t $MODEL_DIR | head -1)

echo -- Evaluating $LATEST_MODEL --

CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=1 --master_port=8723 train_i3d.py \
    --config=$CONFIG \
    --phase=val \
    --ngpu=1 \
    --save-model=$LATEST_MODEL \
    |& tee -a $1
