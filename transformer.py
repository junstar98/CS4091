import tensorflow as tf
import pandas as pd
import re
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, emded_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=emded_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=emded_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

train_data = pd.read_csv("~/ncdata/train.csv")
eval_data = pd.read_csv("~/ncdata/eval.csv")

train_data = train_data[['inputs','targets']]
eval_data = eval_data[['inputs', 'targets']]

train_data.drop_duplicates(subset=['inputs'], inplace=True)
eval_data.drop_duplicates(subset=['inputs'], inplace=True)

train_data.dropna(axis=0, inplace=True)
eval_data.dropna(axis=0, inplace=True)

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

eval_data['inputs'] = clean_eval_text
eval_data['targets'] = clean_eval_summary


train_data.replace('', np.nan, inplace=True)
eval_data.replace('', np.nan, inplace=True)

train_data.dropna(axis = 0, inplace = True)
eval_data.dropna(axis = 0, inplace = True)

#print('텍스트의 최소 길이 : {}'.format(np.min(text_len)))
#print('텍스트의 최대 길이 : {}'.format(np.max(text_len)))
#print('텍스트의 평균 길이 : {}'.format(np.mean(text_len)))
#print('요약의 최소 길이 : {}'.format(np.min(summary_len)))
#print('요약의 최대 길이 : {}'.format(np.max(summary_len)))
#print('요약의 평균 길이 : {}'.format(np.mean(summary_len)))

max_len_text = 25
max_len_summary= 15


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



x_train = list(train_data['inputs'])
y_train = list(train_data['targets'])
x_val = list(eval_data['inputs'])
y_val = list(eval_data['targets'])

vocab_size = 20000
maxlen = 20

embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile('rmsprop', "sparse_categorical_crossentropy")
history = model.fit(
    x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val)
)
