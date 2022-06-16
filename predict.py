import transformers
import torch
import sys
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

torch.set_num_threads(2)
MODEL_NAME = 'xlm-roberta-base'

def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name', default=MODEL_NAME,
                    help='Pretrained model name')
    ap.add_argument('--text', metavar='FILE', required=True,
                    help='Text to be predicted') #could also be string?
    ap.add_argument('--load_model', default=None, metavar='FILE',
                    help='Load model from file')
    ap.add_argument('--threshold', default=0.4, metavar='FLOAT', type=float,
                    help='threshold for calculating f-score')
    ap.add_argument('--labels', choices=['full', 'upper'], default='full')
    ap.add_argument('--output', default=None, metavar='FILE', help='Location to save predictions')
    return ap

options = argparser().parse_args(sys.argv[1:])

labels_full = ['HI', 'ID', 'IN', 'IP', 'LY', 'NA', 'OP', 'SP', 'av', 'ds', 'dtp', 'ed', 'en', 'fi', 'it', 'lt', 'nb', 'ne', 'ob', 'ra', 're', 'rs', 'rv', 'sr']
labels_upper = ['HI', 'ID', 'IN', 'IP', 'LY', 'NA', 'OP', 'SP']

if options.labels == 'full':
    labels = labels_full
else:
    labels = labels_upper
num_labels = len(labels)
#print("Number of labels:", num_labels)

# tokenizer from the pretrained model
tokenizer = transformers.AutoTokenizer.from_pretrained(options.model_name)

# load our fine-tuned model
model = torch.load(options.load_model, map_location=torch.device('cpu'))
#model.to('cpu')

def predict_labels(string):
    tokenized = tokenizer(string, return_tensors='pt')
    pred = model(**tokenized)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(pred.logits.detach().numpy()))
    preds = np.zeros(probs.shape)
    preds[np.where(probs >= options.threshold)] = 1
    return [labels[idx] for idx, label in enumerate(preds.flatten()) if label >= options.threshold]

# predict labels text at a time
with open(options.text, 'r') as f:
    for line in f:
        print(f'{" ".join(predict_labels(line))}')
