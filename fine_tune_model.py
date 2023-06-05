import os 
import math 
import pandas as pd
import argparse
import earlyStopping
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification
import torch
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import torch.nn.functional as F


input_path = 'datasets'
output_path = 'resources'
tokenizer_dir = './tokenizer'
model_dir = './model'
config_dir = './config'

MAX_LEN = 310
TRAIN_BATCH_SIZE = 6
VALID_BATCH_SIZE = 6
MAX_GRAD_NORM = 10
LEARNING_RATE = 1e-5

def read_data(dataset_name):
    train_path = os.path.join(input_path, dataset_name, 'train.txt')
    devel_path = os.path.join(input_path, dataset_name, 'devel.txt')
    train_token_lst, train_label_lst = [], []
    with open(train_path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                train_token_lst.append(math.nan)
                train_label_lst.append(math.nan)
                continue
            a = line.split('\t')
            train_token_lst.append(a[0].strip())
            train_label_lst.append(a[1].strip())

    train_data = pd.DataFrame({'Tokens': train_token_lst, 'Labels': train_label_lst})

    devel_token_lst, devel_label_lst = [], []
    with open(devel_path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                devel_token_lst.append(math.nan)
                devel_label_lst.append(math.nan)
                continue
            a = line.split('\t')
            devel_token_lst.append(a[0].strip())
            devel_label_lst.append(a[1].strip())

    devel_data = pd.DataFrame({'Tokens': devel_token_lst, 'Labels': devel_label_lst})

    return train_data, devel_data

def IdToLabelAndLabeltoId(train_data):
    label_list = train_data["Labels"]
    label_list = [*set(label_list)]
    label_list = [x for x in label_list if not pd.isna(x)]
    # sorting as applying set operation does not maintain the order
    label_list.sort()
    id2label = {}
    for index, label in enumerate(label_list):
        id2label[index] = label
    label2id = { id2label[id]: id for id in id2label}
    return id2label,label2id

def convert_to_sentence(df):
    sent = ""
    sent_list = []
    label = ""
    label_list = []
    for tok,lab in df.itertuples(index = False):
        if isinstance(tok, float):
            sent = sent[1:]
            sent_list.append(sent)
            sent = ""
            label = label[1:]
            label_list.append(label)
            label = ""
        else:
            sent = sent + " " +str(tok)
            label = label+ "," + str(lab)
    if sent != "":
        sent_list.append(sent)
        label_list.append(label)

    return sent_list,label_list

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []
    sentence = str(sentence).strip()
    text_labels = str(text_labels)

    for word, label in zip(sentence.split(), text_labels.split(',')):
        # tokenize and count num of subwords
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        tokenized_sentence.extend(tokenized_word)
        # add same label of word to other subwords
        labels.extend([label]*n_subwords)

    return tokenized_sentence, labels 

class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, label2id, id2label):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer 
        self.max_len = max_len 
        self.label2id = label2id
        self.id2label = id2label

    def __getitem__(self, index):
        # step 1: tokenize sentence and adapt labels
        sentence = self.data.Sentence[index]
        word_labels = self.data.Labels[index]
        label2id = self.label2id

        tokenized_sentence, labels = tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer)

        # step 2: add special tokens and corresponding labels
        tokenized_sentence = ['[CLS]'] + tokenized_sentence + ['[SEP]']
        labels.insert(0, 'O')
        labels.insert(-1, 'O')

        # step 3: truncating or padding
        max_len = self.max_len

        if len(tokenized_sentence) > max_len:
            #truncate
            tokenized_sentence = tokenized_sentence[:max_len]
            labels = labels[:max_len]
        else:
            # pad
            tokenized_sentence = tokenized_sentence + ['[PAD]' for _ in range(max_len - len(tokenized_sentence))]
            labels = labels + ['O' for _ in range(max_len - len(labels))]

        # step 4: obtain attention mask
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]

        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        label_ids = [label2id[label] for label in labels]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            'targets': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return self.len

def train(model, dataloader, optimizer, device):
    tr_loss, tr_accuracy = 0,0
    tr_preds, tr_labels = [], []
    nb_tr_steps = 0
    #put model in training mode
    model.train()

    for idx, batch in enumerate(dataloader):
        ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets  = batch['targets'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels= targets)
        loss, tr_logits = outputs[0], outputs[1]

        tr_loss += loss.item()
        tr_logits = F.softmax(tr_logits, dim=2)
        nb_tr_steps += 1

        if idx % 100 == 0:
            print(f'\ttraining loss at {idx} steps: {tr_loss}')
        #______[See shapes after each operation]______
        #compute training accuracy
        flattened_targets = targets.view(-1)
        # though named logits, is after softmax
        active_logits = tr_logits.view(-1, model.num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)
        # now, use mask to determine where we should compare predictions with targets (includes [CLS] and [SEP] token predictions)
        active_accuracy = mask.view(-1) == 1
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tr_preds.extend(predictions)
        tr_labels.extend(targets)

        tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        #gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters = model.parameters(),
            max_norm = MAX_GRAD_NORM
        )

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss 
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f'\tTraining loss for the epoch: {epoch_loss}')
    print(f'\tTraining accuracy for epoch: {tr_accuracy}')

def valid(model, dataloader, device):
    eval_loss = 0
    nb_eval_steps = 0
    model.eval()

    for idx, batch in enumerate(dataloader):
        ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['targets'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
        loss, eval_logits = outputs[0], outputs[1]

        eval_loss += loss.item()
        eval_logits = F.softmax(eval_logits, dim=2)
        nb_eval_steps += 1

    
    return eval_loss


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--require_training', help='will train model if specified', action='store_true')

    args = parser.parse_args()
    # read data
    train_data, devel_data = read_data(args.dataset_name)
    # get a dict for label and its id
    id2label,label2id = IdToLabelAndLabeltoId(train_data)
    #get list of sentence and associated label
    train_sent, train_label = convert_to_sentence(train_data)
    devel_sent,devel_label = convert_to_sentence(devel_data)

    num_labels = len(id2label)

    # downloading tokenizer
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    # downloading model
    model = AutoModelForTokenClassification.from_pretrained('dmis-lab/biobert-v1.1', num_labels=num_labels, id2label=id2label, label2id=label2id)
    
    device = 'cuda:0' if cuda.is_available() else 'cpu'
    #loading model to device
    model.to(device)

    train_data = {'Sentence':train_sent, 'Labels':train_label}
    train_data = pd.DataFrame(train_data)
    devel_data = {'Sentence':devel_sent, 'Labels':devel_label}
    devel_data = pd.DataFrame(devel_data)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    devel_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    if args.require_training:
        train_dataset = dataset(train_data, tokenizer, MAX_LEN, label2id, id2label)
        train_dataloader = DataLoader(train_dataset, **train_params)
        devel_dataset = dataset(devel_data, tokenizer, MAX_LEN, label2id, id2label)
        devel_dataloader = DataLoader(devel_dataset, **devel_params)
        
        early_stopper = earlyStopping.EarlyStopper(4, 0)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
        num_epochs = 30
        is_saved = False
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}:')
            train(model, train_dataloader, optimizer, device)
            validation_loss = valid(model, devel_dataloader, device)
            print(f'\tValidation loss: {validation_loss}')
            if not is_saved and early_stopper.early_stop(validation_loss):
                print('___________________________________')
                print(f'Early stopped at epoch {epoch+1}.')
                
                model_path = os.path.join(output_path, args.dataset_name, 'model')
                model.save_pretrained(model_path)
                datafile_name = os.path.join(model_path, 'info.txt')
                with open(datafile_name, 'w') as fh:
                    fh.write(f'Dataset: {args.dataset_name}\n')
                    fh.write(f'Epoch: {epoch+1}\n')
                
                tokenizer_path = os.path.join(output_path, args.dataset_name, 'tokenizer')
                tokenizer.save_pretrained(tokenizer_path)
                is_saved = True
        
        model_path = os.path.join(output_path, args.dataset_name, 'model_30epochs')
        model.save_pretrained(model_path)
            



    
if __name__ == '__main__':
    main()
