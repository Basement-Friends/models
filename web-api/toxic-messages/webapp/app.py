import flask
import pickle
import gzip
import re
import spacy
import string
import tensorflow as tf
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def load_tokenizer(filename):
    with open(filename, 'rb') as pickle_file:
        return pickle.load(pickle_file)

def clean_message(message):
    message = message.lower()
    message = re.sub(r"what's", "what is ", message)
    message = re.sub(r"\'s", " ", message)
    message = re.sub(r"\'ve", " have ", message)
    message = re.sub(r"can't", "cannot ", message)
    message = re.sub(r"n't", " not ", message)
    message = re.sub(r"i'm", "i am ", message)
    message = re.sub(r"\'re", " are ", message)
    message = re.sub(r"\'d", " would ", message)
    message = re.sub(r"\'ll", " will ", message)
    message = re.sub(r"\'scuse", " excuse ", message)
    message = re.sub('\W', ' ', message)
    message = re.sub('\s+', ' ', message)
    message = re.sub(r'#[0-9]+|@[0-9a-zA-Z]+|#|https?://[0-9a-zA-Z\./\-_\?]+|â¦|(amp)|[^\x20-\x7e]|â|¥|ð|»|¼|ï|¸|¦|±|¯|[0-9]+', '', message)
    message = message.strip(' ')
    return message

def remove_punctuations_and_digits(text : str):
    to_remove = string.punctuation + string.digits
    cur_text = ''
    for i in range(len(text)):
        if text[i] in to_remove:
            cur_text += ' '
        else:
            cur_text += text[i]
    cur_text = ' '.join(cur_text.split())
    return cur_text

def remove_stop_words(text: str):
    filtered_sentence = []
    for word in text.split(' '):
        lexeme = nlp.vocab[word]
        if lexeme.is_stop == False:
            filtered_sentence.append(word)
    return ' '.join(filtered_sentence)
    
def lemmatize(text : str):
    return ' '.join([token.lemma_ for token in nlp(text)])

app = flask.Flask(__name__, template_folder='templates')
model = load_zipped_pickle('model/toxic_messages_classifier.pkl')
tokenizer = load_tokenizer('model/tokenizer.pkl')
max_len = 1963
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

@app.route('/isToxic', methods=['POST'])
def is_toxic():
    data = flask.request.json
    message = data.get('message')
    message_df = pd.DataFrame(data=list(message), columns =['message'])
    message_df['message'] = message_df['message'].map(lambda message : clean_message(message))
    message_df['message'] = message_df['message'].map(lambda message : remove_punctuations_and_digits(message))
    message_df['message'] = message_df['message'].map(lambda message : remove_stop_words(message))
    message_df['message'] = message_df['message'].map(lambda message : lemmatize(message))
    X = tokenizer.texts_to_sequences(message_df['message'])
    X = pad_sequences(X, padding='post', maxlen=max_len)
    prediction = model.predict(X)
    return str(int(prediction[0][1]))

if __name__ == '__main__': 
    app.run()