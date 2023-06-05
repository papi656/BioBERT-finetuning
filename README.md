### `download_model.py`
- This file downloads and saves locally the biobert-v1.1 model and tokenizer from huggingface. 

### `fine_tune_model.py`

`def read_data(dataset_name)`
- parameter:
    - **dataset_name**: Takes dataset name (eg: BC5CDR, MedMentions, NCBI-disease) to get location of training and development set.
- return value:
    - two dataframes train_data and devel_data, with two columns-['Tokens','Labels']
- description:
    - Reads the BIO tagged data from .txt file and convert it into a dataframe

`def IdToLabelAndLabeltoId(train_data)`
- parameter:
    - **train_data**: takes the train_data dataframe return from `read_data(...)`
- return value:
    - **id2label,label2id**: two dictionaries which maps labels as follows `B:0, I:1, O:2`
- description:
    - Gives dictionaries to convert labels to id and vice-versa.


`def convert_to_sentence(df)`
- parameter:
    - **df**: dataframe returned by `read_data(...)`
- return value:
    - **sent_lst, label_lst**: one list containing sentence and other containing corresponding labels
- description:
    - Given a dataframe having tokens and labels, it constructs a sentence by joing a token until a blank space is reached. Labels are also joined accordingly.

`[template]`
- parameter:
    - tmp
- return value:
    - tmp
- description:
    - tmp