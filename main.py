# libraries
import string
import pandas as pd
import tensorflow as tf
import numpy as np
import re
from keras.models import load_model
from tensorflow.keras import layers , activations , models , preprocessing
import discord
import os

# set random seed to match seed used for model training
tf.random.set_seed(69)

csv_name = 'topical_chat.csv'
df = pd.read_csv(csv_name, index_col=False)
df.drop('conversation_id', axis=1, inplace=True)
df.drop('sentiment', axis=1, inplace=True)

questions = df[0:5000:2].reset_index(drop=True)
responses = df[1:5000:2].reset_index(drop=True)

data = pd.DataFrame()
data['questions'] = questions
data['answers'] = responses
data = data.sample(frac=1, random_state=69).reset_index(drop=True)

pairs = list(zip(data['questions'], data['answers']))

# clean_text function: strips apostrophes, contractions
def clean_text(text):
  text = str(text).lower()
  text = text.replace("can't", "cannot")
  text = text.replace("won't", "will not")
  text = text.replace("i'm", "i am")
  text = text.replace("'ve", " have")
  text = text.replace("'ll", " will")
  text = text.replace("n't", " not")
  text = text.replace("'d", " would")
  text = text.replace("'s", " is")
  text = text.replace("'re", " are")
  for char in string.punctuation:
    if char != '-':
      text = text.replace(char, '')
  return text


input_docs = []
target_docs = []
input_tokens = set()
target_tokens = set()

for line in pairs[:1000]:
  input_doc, target_doc = line[0], line[1]

  input_doc = clean_text(input_doc)
  target_doc = clean_text(target_doc)

  # Appending each input sentence to input_docs
  input_docs.append(input_doc)

  # Splitting words from punctuation
  target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))

  # Redefine target_doc below and append it to target_docs
  target_doc = '<START> ' + target_doc + ' <END>'
  target_docs.append(target_doc)

  # Now we split up each sentence into words and add each unique word to our vocabulary set
  for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
    if token not in input_tokens:
      input_tokens.add(token)

  for token in target_doc.split():
    if token not in target_tokens:
      target_tokens.add(token)

input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

input_features_dict = dict([(token, i)
                            for i, token in enumerate(input_tokens)])
target_features_dict = dict([(token, i)
                             for i, token in enumerate(target_tokens)])

reverse_input_features_dict = dict(
  (i, token) for token, i in input_features_dict.items())
reverse_target_features_dict = dict(
  (i, token) for token, i in target_features_dict.items())

max_encoder_seq_length = max(
  [len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
max_decoder_seq_length = max([
  len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs
])


def str_to_tokens(sentence: str):
  words = sentence.lower().split()
  tokens_list = list()
  for word in words:
    if word in input_features_dict:
      tokens_list.append(input_features_dict[word])
  return preprocessing.sequence.pad_sequences([tokens_list],
                                              maxlen=max_encoder_seq_length,
                                              padding='post')


enc_model = load_model('enc_model.h5')
dec_model = load_model('dec_model.h5')

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

def help_msg():
  return "Type '$tb ' followed by desired message to communicate with Terminator bot!"


#Generate AI bot response
def gen_msg(input_command):
  states_values = enc_model.predict(str_to_tokens(input_command), verbose=0)
  empty_target_seq = np.zeros((1, 1))
  empty_target_seq[0, 0] = target_features_dict['<START>']
  stop_condition = False
  decoded_translation = ''
  while not stop_condition:
    dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values,
                                          verbose=0)
    sampled_word_index = np.argmax(dec_outputs[0, -1, :])
    sampled_word = reverse_target_features_dict[sampled_word_index]

    if sampled_word == '<END>' or len(
        decoded_translation.split()) > max_decoder_seq_length:
      stop_condition = True
      continue

    decoded_translation += " " + sampled_word

    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = sampled_word_index
    states_values = [h, c]
  return decoded_translation

#Get rid of '$tb '
def filter_input(input_msg):
  return input_msg[4::]


@client.event
async def on_ready():
  print('We have logged in as {0.user}'.format(client))


@client.event
async def on_message(message):

  if message.author == client.user:
    return

  if message.content.startswith('$tb '):
    out_msg = gen_msg(filter_input(message.content))
    await message.channel.send(out_msg)

  if message.content.startswith('$help'):
    await message.channel.send(help_msg())


# HERE: client.run(YOUR_DISCORD_TOKEN)
