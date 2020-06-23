import numpy as np  
import pandas as pd 
import re           
import nltk
from attention import AttentionLayer
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords   
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

train_data = pd.read_csv("~/ncdata/train.csv")
print(len(train_data))
eval_data = pd.read_csv("~/ncdata/eval.csv")

eval_data = eval_data[['inputs', 'targets']]
train_data = train_data[['inputs','targets']]


#train_data2 = pd.read_csv("result.csv", names=['inputs', 'targets', 'similarity'])
#train_data2 = train_data2[['inputs', 'targets']]

#train_data += train_data2

train_data.drop_duplicates(subset=['inputs'], inplace=True)
eval_data.drop_duplicates(subset=['inputs'], inplace=True)

train_data.dropna(axis=0, inplace=True)
eval_data.dropna(axis=0, inplace=True)

#stop_words = set(['아', '휴', '아이구', '어', '나', '아이쿠', '아이고', '우리', '저희', '따라', '의해', '을', '를', '에', '의', '가', '으로', '로', '에게', '뿐이다', '의거하여', '근거하여', '입각하여', '기준으로', '예하면', '예를 들면', '예를 들자면', '저', '저희', '지말고', '소인', '하지마', '하지마라', '다른', '물론', '또한', '그리고', '해서는 안된다', '뿐만아니라', '만이 아니다', '만은 아니다', '막론하고', '관계없이', '그러나', '그런데', '하지만', '설사', '비록', '더라도', '아니면', '만 못하다', '하는 편이 낫다', '불문하고', '향하여', '향해서', '향하다', '쪽으로', '틈타', '이용하여', '타다', '오르다', '제외하고', '이 외에', '이 밖에', '하여야', '비로소', '외에도', '이곳', '여기', '부터', '따라서', '할 생각이다', '하려고하다', '이리하여', '그리하여', '하지만', '일때', '할때', '로써', '으로써', '까지', '해야한다', '일것이다', '반드시', '한다면', '한다면', '등', '등등', '제', '겨우', '단지', '다만', '할뿐', '대해서', '대하여', '대하면', '훨씬', '얼마나', '얼마만큼', '얼마큼', '약간', '그래도', '그리고', '너', '너희들', '타인', '때문에', '이젠', '만큼', '지든지', '몇', '거의', '가령', '설령', '지든지', '즉', '및', '혹시', '혹은', '설령', '가령', '동시에', '공동으로', '위하여', '것들', '것', '타인', '너희들', '그들', '그', '때문에'])
stop_words = set()
def preprocess_sentence(sentence, remove_stopwords = True):
    sentence = re.sub(r'\([^)]*\)', '', sentence) # 괄호로 닫힌 문자열  제거 Ex) my husband (and myself) for => my husband for
    sentence = re.sub('"','', sentence) # 쌍따옴표 " 제거
    sentence = re.sub("[^가-힣]", " ", sentence) # 영어 외 문자(숫자, 특수문자 >등) 공백으로 변환
    if remove_stopwords:
        tokens = ' '.join(word for word in sentence.split() if not word in stop_words if len(word) > 1)
    else:
        tokens = ' '.join(word for word in sentence.split() if len(word) > 1)
    return tokens


clean_train_text = []
for s in train_data['inputs']:
    clean_train_text.append(preprocess_sentence(s))


clean_train_summary = []
for s in train_data['targets']:
    clean_train_summary.append(preprocess_sentence(s, 0))

clean_eval_text = []
for s in eval_data['inputs']:
    clean_eval_text.append(preprocess_sentence(s))


clean_eval_summary = []
for s in eval_data['targets']:
    clean_eval_summary.append(preprocess_sentence(s, 0))


train_data['inputs'] = clean_train_text
train_data['targets'] = clean_train_summary

train_data.replace('', np.nan, inplace=True)

train_data.dropna(axis = 0, inplace = True)

eval_data['inputs'] = clean_eval_text
eval_data['targets'] = clean_eval_summary

eval_data.replace('', np.nan, inplace=True)

eval_data.dropna(axis = 0, inplace = True)


#print('텍스트의 최소 길이 : {}'.format(np.min(text_len)))
#print('텍스트의 최대 길이 : {}'.format(np.max(text_len)))
#print('텍스트의 평균 길이 : {}'.format(np.mean(text_len)))
#print('요약의 최소 길이 : {}'.format(np.min(summary_len)))
#print('요약의 최대 길이 : {}'.format(np.max(summary_len)))
#print('요약의 평균 길이 : {}'.format(np.mean(summary_len)))

max_len_text = 25
max_len_summary= 15

from sklearn.model_selection import train_test_split


def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s.split()) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))))

#below_threshold_len(text_max_len, train_data['inputs'])
#below_threshold_len(summary_max_len, train_data['targets'])

train_data = train_data[train_data['inputs'].apply(lambda x: len(x.split()) <= max_len_text)]
train_data = train_data[train_data['targets'].apply(lambda x: len(x.split()) <= max_len_summary)]

eval_data = eval_data[eval_data['inputs'].apply(lambda x: len(x.split()) <= max_len_text)]
eval_data = eval_data[eval_data['targets'].apply(lambda x: len(x.split()) <= max_len_summary)]



train_data['targets'] = train_data['targets'].apply(lambda x : 'sostoken '+ x + ' eostoken')

eval_data['targets'] = eval_data['targets'].apply(lambda x : 'sostoken '+ x + ' eostoken')

x_tr,x_val,y_tr,y_val=train_test_split(train_data['inputs'],train_data['targets'],test_size=0.1,random_state=0,shuffle=True) 

x_eval = eval_data['inputs']
y_eval = eval_data['targets']


x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(x_tr))

#convert text sequences into integer sequences
x_tr    =   x_tokenizer.texts_to_sequences(x_tr) 
x_val   =   x_tokenizer.texts_to_sequences(x_val)
x_eval  =   x_tokenizer.texts_to_sequences(x_eval)



#padding zero upto maximum length
x_tr    =   pad_sequences(x_tr,  maxlen=max_len_text, padding='post') 
x_val   =   pad_sequences(x_val, maxlen=max_len_text, padding='post')
x_eval   =   pad_sequences(x_eval, maxlen=max_len_text, padding='post')

x_voc_size   =  len(x_tokenizer.word_index) +1

y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(y_tr))

#convert summary sequences into integer sequences
y_tr    =   y_tokenizer.texts_to_sequences(y_tr)
y_val   =   y_tokenizer.texts_to_sequences(y_val)
y_eval  =  y_tokenizer.texts_to_sequences(y_eval)

#padding zero upto maximum length
y_tr    =   pad_sequences(y_tr, maxlen=max_len_summary, padding='post')
y_val   =   pad_sequences(y_val, maxlen=max_len_summary, padding='post')
y_eval  =  pad_sequences(y_eval, maxlen=max_len_summary, padding='post')

y_voc_size  =   len(y_tokenizer.word_index) +1

from tensorflow.keras import backend as K
K.clear_session()
latent_dim = 700

# Encoder
encoder_inputs = Input(shape=(max_len_text,))
enc_emb = Embedding(x_voc_size, latent_dim,trainable=True)(encoder_inputs)

#LSTM 1
encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

#LSTM 2
encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

#LSTM 3
encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True)
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

# Set up the decoder.
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(y_voc_size, latent_dim,trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

#LSTM using encoder_states as initial state
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

#Attention Layer

attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

# Concat attention output and decoder LSTM output
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#Dense layer
decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

history=model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:] ,epochs=50, batch_size=1024, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))

model.save('basic_learning_model5')



reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index

# encoder inference
encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

encoder_model.save('basic_encoder_model5')

# decoder inference
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_len_text,latent_dim))

# Get the embeddings of the decoder sequence
dec_emb2= dec_emb_layer(decoder_inputs)

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])


#attention inference
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_inf_concat)

# Final decoder model
decoder_model = Model(
[decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
[decoder_outputs2] + [state_h2, state_c2])

decoder_model.save('basic_decoder_model5')

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Chose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = target_word_index['sostoken']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if(sampled_token!=''):
            # Exit condition: either hit max length or find stop word.
            if (sampled_token == 'eostoken' or len(decoded_sentence.split()) >= (max_len_summary-1)):
                stop_condition = True
            else:
                decoded_sentence += ' '+sampled_token

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence

def seq2summary(input_seq):
    newString=''
    for i in input_seq:
      if((i!=0 and i!=target_word_index['sostoken']) and i!=target_word_index['eostoken']):
        newString=newString+reverse_target_word_index[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
      if(i!=0):
        newString=newString+reverse_source_word_index[i]+' '
    return newString

bleu_score = 0
print("Computing Bleu Score...")
for i in range(len(x_eval)):
  print(i)
  bleu_score += nltk.translate.bleu_score.sentence_bleu([seq2summary(y_eval[i]).strip().split()], decode_sequence(x_eval[i].reshape(1, max_len_text)).strip().split())

bleu_score /= len(x_eval)
bleu_score *= 100;

print("Bleu Score =", bleu_score)
