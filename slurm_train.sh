#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH -p gpu
#SBATCH -t 01:15:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=Project_2002026
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

echo "START: $(date)"

rm logs/current.err
rm logs/current.out
ln -s $SLURM_JOBID.err logs/current.err
ln -s $SLURM_JOBID.out logs/current.out

module purge
module load pytorch
source venv/bin/activate
export PYTHONPATH=venv/lib/python3.8/site-packages:$PYTHONPATH
export TRANSFORMERS_CACHE=cachedir
pip3 install --upgrade pip
pip3 install transformers
#pip3 install datasets

MODEL="xlm-roberta-base"
#MODEL="xlm-roberta-large"
#MODEL="TurkuNLP/bert-base-finnish-cased-v1"
#MODEL="KB/bert-base-swedish-cased"
#MODEL="camembert-base"
MODEL_ALIAS="xlmr-base"
SRC=$1
TRG=$2
LR_=$3
EPOCHS_=$4
i=$5
BS=7

echo "MODEL:$MODEL"
echo "MODEL_ALIAS:$MODEL_ALIAS"
echo "SRC:$SRC"
echo "TRG:$TRG"
echo "LR:$LR_"
echo "EPOCHS:$EPOCHS_"
echo "i:$i"

export TRAIN_DIR=data/$SRC
export TEST_DIR=data/$TRG
export OUTPUT_DIR=output

mkdir -p "$OUTPUT_DIR"

for EPOCHS in $EPOCHS_; do
for LR in $LR_; do
for j in $i; do
#rm -r checkpoints/$MODEL_ALIAS-$SRC-$TRG-$LR/
echo "Settings: src=$SRC trg=$TRG model=$MODEL lr=$LR epochs=$EPOCHS batch_size=$BS"
echo "job=$SLURM_JOBID src=$SRC trg=$TRG model=$MODEL lr=$LR epochs=$EPOCHS batch_size=$BS" >> logs/experiments.log
srun python train.py \
  --model_name $MODEL \
  --train $TRAIN_DIR/train.tsv \
  --dev $TRAIN_DIR/dev.tsv \
  --test $TEST_DIR/test.tsv \
  --lr $LR \
  --epochs $EPOCHS \
  --batch_size $BS \
  --checkpoints checkpoints/$MODEL_ALIAS-$SRC-$TRG-$LR \
  --labels full #\
#  --save_model models/$MODEL_ALIAS-$SRC-$TRG.pt
# --threshold 0.4

#rm -r checkpoints/$MODEL_ALIAS-$SRC-$TRG-$LR/

done
done
done

echo "job=$SLURM_JOBID src=$SRC trg=$TRG model=$MODEL lr=$LR epochs=$EPOCHS batch_size=$BS" >> logs/completed.log

echo "END: $(date)"
