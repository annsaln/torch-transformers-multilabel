# torch-transformers-multilabel

This codebase was built with PyTorch 1.11, Transformers 4.18.0 and Hugging Face datasets. 

To train a model, run

    python3 train.py --train train.tsv --dev dev.tsv --test test.tsv
  
With the slurm script, you can launch multiple instances:

    sbatch slurm_train_arg.sh [TRAIN_DATA] [TEST_DATA] [LRs] [EPOCHSs] [INSTANCEs]
