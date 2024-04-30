import streamlit as st
import pickle
import time
import numpy as np
import tensorflow

import tensorflow.keras as keras

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
from keras.preprocessing.sequence import pad_sequences

model = pickle.load(open('model.pkl','rb'))
def processing():
    with open('IndiaUS.txt', 'r') as file:
        faqs = file.read()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([faqs])

    word_len = len(tokenizer.word_index) + 1

    input_sequences = []
    max_len = 0
    for sentence in faqs.split('\n'):
        tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]

        for i in range(1, len(tokenized_sentence)):
            input_sequences.append(tokenized_sentence[:i + 1])

    max_len = max([len(x) for x in input_sequences])
    return max_len
def find_similar(text,max_len):
    result=[]
    for i in range(10):
        # tokenize
        token_text = tokenizer.texts_to_sequences([text])[0]
        # padding
        padded_token_text = pad_sequences([token_text], maxlen=max_len - 1, padding='pre')
        # predict
        pos = np.argmax(model.predict(padded_token_text))

        for word, index in tokenizer.word_index.items():
            if index == pos:
                text = text + " " + word
                result.append(text)
                time.sleep(2)
    return result

def main():
    st.title('Similar Words')

    input_data=st.text_input('Enter the word')
    max_len = processing()
    result = []
    if st.button('Find'):
        result=find_similar(input_data,max_len)
        for sentence in result:
            st.write(sentence)  # Use st.text() instead of print()

    # st.success(result)

if __name__ == '__main__':  # Corrected from '_main_'
    main()


