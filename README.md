# torch-transformers-multilabel

This codebase was built with PyTorch 1.11, Transformers 4.18.0 and Hugging Face datasets. 

To train a model, run

    python3 train.py --train [TRAIN_LANGUAGE] --dev [TRAIN_LANGUAGE] --test [TEST_LANGUAGE]
    
Use several languages in training or testing by connecting them with a hyphen. E.g., en-fi-sv.
  
With the slurm script, you can launch multiple instances:

    sbatch slurm_train.sh [TRAIN_LANGUAGE] [TEST_LANGUAGE] [LRs] [EPOCHSs] [INSTANCEs]

To predict labels for a text file (several texts when separated by linebreak):
    
    python3 predict.py --text text.txt --load_model model.pt

List of registers and their abbreviations (used as labels):

| Register                           | Label         |
|------------------------------------|--------------|
| How-to                             | HI           |
| Interactive Discussion             | ID           |
| Informative                      | IN           |
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
