CONFIG="configs/gunshot.json" 
GPUS=1,2,3,4,5,6
NUM_GPUS=6

if [ $# -ne 1 ]; then
    echo Bad arguments
    exit
fi

mkdir -p $(dirname $1)

CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS train_i3d.py \
    --name=$1 \
    --ngpu=$NUM_GPUS \
    --config=$CONFIG \
    --phase="train" \
    |& tee -a $1
