import transformers
import datasets
import torch
import logging
import sys
import numpy as np
logging.disable(logging.INFO)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support,  roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import EarlyStoppingCallback
from pprint import PrettyPrinter
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

pprint = PrettyPrinter(compact=True).pprint
# default arguments
LEARNING_RATE=1e-4
BATCH_SIZE=8
TRAIN_EPOCHS=2
MODEL_NAME = 'xlm-roberta-base'
PATIENCE = 5

# omit av, ed, fi
#labels_full = ['HI', 'ID', 'IN', 'IP', 'LY', 'NA', 'OP', 'SP', 'ds', 'dtp', 'en', 'it', 'lt', 'nb', 'ne', 'ob', 'ra', 're', 'rs', 'rv', 'sr']

labels_full = ['HI', 'ID', 'IN', 'IP', 'LY', 'NA', 'OP', 'SP', 'av', 'ds', 'dtp', 'ed', 'en', 'fi', 'it', 'lt', 'nb', 'ne', 'ob', 'ra', 're', 'rs', 'rv', 'sr']
labels_upper = ['HI', 'ID', 'IN', 'IP', 'LY', 'NA', 'OP', 'SP']

def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name', default=MODEL_NAME,
                    help='Pretrained model name')
    ap.add_argument('--train', required=True,
                    help='Path to training data')
    ap.add_argument('--dev', required=True,
                    help='Path to validation data')
    ap.add_argument('--test', required=True,
                    help='Path to test data')
    ap.add_argument('--batch_size', metavar='INT', type=int,
                    default=BATCH_SIZE,
                    help='Batch size for training')
    ap.add_argument('--epochs', metavar='INT', type=int, default=TRAIN_EPOCHS,
                    help='Number of training epochs')
    ap.add_argument('--lr', '--learning_rate', metavar='FLOAT', type=float,
                    default=LEARNING_RATE, help='Learning rate')
    ap.add_argument('--patience', metavar='INT', type=int,
                    default=PATIENCE, help='Early stopping patience')
    ap.add_argument('--checkpoints', default='checkpoints', metavar='FILE',
                    help='Save model checkpoints to directory')
    ap.add_argument('--save_model', default=None, metavar='FILE',
                    help='Save model to file')
    ap.add_argument('--threshold', default=None, metavar='FLOAT', type=float,
                    help='threshold for calculating f-score')
    ap.add_argument('--labels', choices=['full', 'upper'], default='full')
    ap.add_argument('--load_model', default=None, metavar='FILE',
                    help='load existing model')
    ap.add_argument('--class_weights', default=False, type=bool)
    #ap.add_argument('--save_predictions', default=False, action='store_true',
    #                help='save predictions and labels for dev set, or for test set if provided')
    return ap


options = argparser().parse_args(sys.argv[1:])
if options.labels == 'full':
    labels = labels_full
else:
    labels = labels_upper
num_labels = len(labels)
print("Number of labels:", num_labels)

#register scheme mapping:
sub_register_map = {
    'NA': 'NA',
    'NE': 'ne',
    'SR': 'sr',
    'PB': 'nb',
    'HA': 'NA',
    'FC': 'NA',
    'TB': 'nb',
    'CB': 'nb',
    'OA': 'NA',
    'OP': 'OP',
    'OB': 'ob',
    'RV': 'rv',
    'RS': 'rs',
    'AV': 'av',
    'IN': 'IN',
    'JD': 'IN',
    'FA': 'fi',
    'DT': 'dtp',
    'IB': 'IN',
    'DP': 'dtp',
    'RA': 'ra',
    'LT': 'lt',
    'CM': 'IN',
    'EN': 'en',
    'RP': 'IN',
    'ID': 'ID',
    'DF': 'ID',
    'QA': 'ID',
    'HI': 'HI',
    'RE': 're',
    'IP': 'IP',
    'DS': 'ds',
    'EB': 'ed',
    'ED': 'ed',
    'LY': 'LY',
    'PO': 'LY',
    'SO': 'LY',
    'SP': 'SP',
    'IT': 'it',
    'FS': 'SP',
    'TV': 'SP',
    'OS': 'OS',
    'IG': 'IP',
    'MT': 'MT',
    'HT': 'HI',
    'FI': 'fi',
    'OI': 'IN',
    'TR': 'IN',
    'AD': 'OP',
    'LE': 'OP',
    'OO': 'OP',
    'MA': 'NA',
    'ON': 'NA',
    'SS': 'NA',
    'OE': 'IP',
    'PA': 'IP',
    'OF': 'ID',
    'RR': 'ID',
    'FH': 'HI',
    'OH': 'HI',
    'TS': 'HI',
    'OL': 'LY',
    'PR': 'LY',
    'SL': 'LY',
    'TA': 'SP',
    'OTHER': 'OS',
    '': '',
#    'av': 'OP', # uncomment if you want to combine these into upper registers
#    'ed': 'IP',
#    'fi': 'IN'
}

def remove_NA(d):
  """ Remove null values and separate multilabel values with comma """
  if d['label'] == None:
    d['label'] = 'NA'
  if ' ' in d['label']:
    d['label'] = ",".join(sorted(d['label'].split()))
  return d

def label_encoding(d):
  """ Split the multi-labels """
  d['label'] = d['label'].split(",")
  mapped = [sub_register_map[l] if l not in labels else l for l in d['label']]
  d['label'] = np.array(sorted(list(set(mapped))))
  return d

def binarize(dataset):
    """ Binarize the labels of the data. Fitting based on the whole data. """
    mlb = MultiLabelBinarizer()
    mlb.fit([labels])
    print("Binarizing the labels")
    dataset = dataset.map(lambda line: {'label': mlb.transform([line['label']])[0]})
    return dataset

data_files = {'train': [], 'dev':[], 'test':[]}

# only train and test for these languages
small_languages = ['ar', 'ca', 'es', 'fa', 'hi', 'id', 'jp', 'no', 'pt', 'tr', 'ur', 'zh']

# choose data with all languages with option 'multi'
for l in options.train.split('-'):
    data_files['train'].append(f'data/{l}/train.tsv')
    if not (l == 'multi' or l in small_languages): 
        data_files['dev'].append(f'data/{l}/dev.tsv')
for l in options.test.split('-'):
    # check if zero-shot for small languages, if yes then test with full data
    if l in small_languages and not (l in options.train.split('-') or 'multi' in options.train.split('-')):
        data_files['test'].append(f'data/{l}/{l}.tsv')
    else:
        data_files['test'].append(f'data/{l}/test.tsv')

dataset = datasets.load_dataset(
    "csv", 
    data_files=data_files, #{'train':options.train, 'test':options.test, 'dev': options.dev}, 
    delimiter="\t",
    column_names=['label', 'text'],
    features=datasets.Features({    # Here we tell how to interpret the attributes
      "text":datasets.Value("string"),
#      "label":datasets.Value("int32")
      "label":datasets.Value("string")}),
    cache_dir = "cachedir"
    )
dataset = dataset.shuffle(seed=42)

#smaller tests
#dataset["train"]=dataset["train"].select(range(400))
#dataset["dev"]=dataset["dev"].select(range(100))
#pprint(dataset['test']['label'][:10])
dataset = dataset.map(remove_NA)
# remove examples that have more than four labels
dataset = dataset.filter(lambda example: len(example['label'].split(','))<=4)
dataset = dataset.filter(lambda example: 'MT' not in example['label'].split(',') and 'OS' not in example['label'].split(','))
dataset = dataset.map(label_encoding)

def compute_class_weights(dataset):
    freqs = [0] * len(labels)
    n_examples = len(dataset['train'])
    for e in dataset['train']['label']:
        for i in range(len(labels)):
            if e[i] != 0:
                freqs[i] += 1
    weights = []
    for i in range(len(labels)):#, label in enumerate(labels):
        weights.append(n_examples/(len(labels)*freqs[i]))
    print("weights:", weights)
    class_weights = torch.FloatTensor(weights).cuda()
    return class_weights

#class_weights = compute_class_weights(dataset)
    
dataset = binarize(dataset)
#pprint(dataset['test']['label'][:5])
if options.class_weights is True:
    class_weights = compute_class_weights(dataset)

model_name = options.model_name #"xlm-roberta-base"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
#        padding=True,
#        return_tensors='pt'
    )

# Apply the tokenizer to the whole dataset using .map()
#dataset = dataset.map(tokenize)
#print(dataset['test'][0])

# evaluate only
if options.load_model is not None:
    model = torch.load(options.load_model)
#    torch.device
    model.to('cpu')
    trues = dataset['test']['label']
    inputs = dataset['test']['text']
    pred_labels = []    
    for index, i in enumerate(inputs):
        tok = tokenizer(i, truncation=True, max_length=512, return_tensors='pt')
        pred = model(**tok)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(pred.logits.detach().numpy()))
        preds = np.zeros(probs.shape)
        preds[np.where(probs >= options.threshold)] = 1
        pred_labels.extend(preds)
#        print("preds",[labels[idx] for idx, label in enumerate(preds.flatten()) if label >= options.threshold])
#        print("trues",[labels[idx] for idx, label in enumerate(trues[index]) if label >= options.threshold])
#        print(i)
#    print(pred_labels)
#    print(trues)
    print("F1-score", f1_score(y_true=trues, y_pred=pred_labels, average='micro'))
    print(classification_report(trues, pred_labels, target_names=labels))
    sys.exit()
    #return [labels[idx] for idx, label in enumerate(preds) if label >= options.threshold]

# Apply the tokenizer to the whole dataset using .map()
dataset = dataset.map(tokenize)

#set up a separated directory for caching
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir="cachedir/")

class MultilabelTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if options.class_weights == True:
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight = class_weights)
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss

print("Model type: ", options.model_name)
print("Learning rate: ", options.lr)
print("Batch size: ", options.batch_size)
print("Epochs: ", options.epochs)

trainer_args = transformers.TrainingArguments(
    options.checkpoints,
    evaluation_strategy="epoch",
    save_strategy='epoch',
    logging_strategy="epoch",
    load_best_model_at_end=True,
    eval_steps=100,
    logging_steps=100,
    learning_rate=options.lr,#0.000005,#0.000005
    per_device_train_batch_size=options.batch_size,
    per_device_eval_batch_size=32,
    num_train_epochs=options.epochs,
#    max_steps=1000,
)

data_collator = transformers.DataCollatorWithPadding(tokenizer)

# Argument gives the number of steps of patience before early stopping
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=5
)

threshold = options.threshold

# in case a threshold was not given, choose the one that works best with the evaluated data
def optimize_threshold(predictions, labels):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    best_f1 = 0
    best_f1_threshold = 0.5 # use 0.5 as a default threshold
    y_true = labels
    for th in np.arange(0.3, 0.7, 0.05):
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= th)] = 1
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = th
    return best_f1_threshold 

def multi_label_metrics(predictions, labels, threshold):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_th05 = np.zeros(probs.shape)
    y_th05[np.where(probs >= 0.5)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average, # user-chosen or optimized threshold
               'f1_th05': f1_score(y_true=y_true, y_pred=y_th05, average='micro'), # report also f1-score with threshold 0.5
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    if options.threshold == None:
        best_f1_th = optimize_threshold(preds, p.label_ids)
        threshold = best_f1_th
        print("Best threshold:", threshold)
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids,
        threshold=threshold)
    return result

class LogSavingCallback(transformers.TrainerCallback):
    def on_train_begin(self, *args, **kwargs):
        self.logs = defaultdict(list)
        self.training = True

    def on_train_end(self, *args, **kwargs):
        self.training = False

    def on_log(self, args, state, control, logs, model=None, **kwargs):
        if self.training:
            for k, v in logs.items():
                if k != "epoch" or v not in self.logs[k]:
                    self.logs[k].append(v)

training_logs = LogSavingCallback()
threshold = options.threshold

trainer = None
trainer = MultilabelTrainer(
    model=model,
    args=trainer_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer = tokenizer,
    callbacks=[early_stopping, training_logs]
)
print("Training...")
trainer.train()

print("Evaluating with test set...")
eval_results = trainer.evaluate(dataset["test"])

pprint(eval_results)

test_pred = trainer.predict(dataset['test'])
trues = test_pred.label_ids
predictions = test_pred.predictions
#print("true:")
#print(trues)
if threshold == None:
    threshold = optimize_threshold(predictions, trues)
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(torch.Tensor(predictions))
# next, use threshold to turn them into integer predictions
preds = np.zeros(probs.shape)
preds[np.where(probs >= threshold)] = 1

# if you want to check the predictions
#for i, (t, p) in enumerate(zip(trues,preds)):
#  print("true", [labels[idx] for idx, label in enumerate(t) if label == 1])
#  print("pred", [labels[idx] for idx, label in enumerate(p) if label > threshold])
#  print(dataset['test']['text'][i])


print(classification_report(trues, preds, target_names=labels))

if options.save_model is not None:
    torch.save(trainer.model, options.save_model)
