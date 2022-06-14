# torch-transformers-multilabel

This codebase was built with PyTorch 1.11, Transformers 4.18.0 and Hugging Face datasets. 

To train a model, run

    python3 train.py --train train.tsv --dev dev.tsv --test test.tsv
  
With the slurm script, you can launch multiple instances:

    sbatch slurm_train_arg.sh [TRAIN_DATA] [TEST_DATA] [LRs] [EPOCHSs] [INSTANCEs]

List of registers and their abbreviations (used as labels):

| Register                           | Label         |
|------------------------------------|--------------|
| How-to                             | HI           |
| Interactive Discussion             | ID           |
| Informational                      | IN           |
| Informational Persuasion           | IP           |
| Lyrical                            | LY           |
| Narrative                          | NA           |
| Opinion                            | OP           |
| Spoken                             | SP           |
| Advice                             | av           |
| Description with intent to sell    | ds           |
| Description of a thing or a person | dtp          |
| News & opinion blog or editorial   | ed           |
| Encyclopedia article               | en           |
| FAQ                                | fi           |
| Interview                          | it           |
| Legal terms and conditions         | lt           |
| Narrative blog                     | nb           |
| News report                        | ne           |
| Opinion blo                        | ob           |
| Research article                   | ra           |
| Recipe                             | re           |
| Denominational religious blog / sermon | rs           |
| Review                             | rv           |
| Sports report                      | sr           |
