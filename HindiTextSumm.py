import streamlit as st
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd 
import torch
import random
from keras.preprocessing.sequence import pad_sequences
import keras
import tensorflow as tf
import json
from keras_preprocessing.text import tokenizer_from_json
import re
import numpy as np
import googletrans
from googletrans import Translator
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers import BartTokenizer, BartForConditionalGeneration

st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'>Hindi Text Summarizer</h1>", unsafe_allow_html=True)

st.sidebar.title("Settings")

with st.sidebar:
    st.markdown("<i style='color: #808080;'>Select the model for summarizing the text.\nYou can also select the maximum no. of words present in the summary by selecting the value through the slider.</i>", unsafe_allow_html=True)
    models = ['LSTM',
        'PEGASUS',
        'T5',
        'BART'
        ]
    model = st.selectbox(label="Summarization Models:", options = models)
    st.markdown("---")
    max_words = st.slider('Max words in the summary:', 50, 500, 350, 50)
    st.markdown("---")
col1, col2, col3, col4, col5 = st.columns([1,2.5,1,2.5,1])
with col2:
    input_text = st.text_area('Text to summarize:', '''
    दिल्ली के कई इलाकों में तेज बारिश से पानी जमा हो गया है। कुछ जगहों से जाम लगने की खबर भी आ रही है। बारिश से पहले दिन में ही अंधेरा छा गया। एयरपोर्ट के टर्मिनल पर पानी भर गया है। हालांकि इसका विमानों की आवाजाही पर कोई असर अभी तक नहीं पड़ा है। उधर चंडीगढ़ में आज सुबह हुई भारी बारिश से कई इलाक़ों में पानी भर गया। तेज बारिश से कई पेड़ गए और कई जगह बिजली के खंभे भी गिर गए। जिसकी वजह से कई सेक्टरों की बिजली गुल हो गई। बारिश की वजह से सड़कों पर पानी भर गया और कई घंटे तक जाम लगा रहा। बिजली गुल होने से ट्रैफिक लाइट्स भी बंद हो गई जिससे चौराहों पर ट्रैफिक को मैनेज करने में काफी परेशानी हुई। भारी बारिश की वजह से कई स्कूलों में छुट्टी भी करनी पड़ी। मौसम विभाग के मुताबिक अगले एक दो दिन इसी तरह की तेज बारिश होने की उम्मीद हैं। वहीं पंजाब के जालंधर में तेज बारिश से कई इलाके पूरी तरह पानी में डूब गए। जालंधर बस अड्डे से लेकर बाजारों तक में पानी भर गया। तेज बारिश की वजह से दोपहर तक बाजार बंद रहे। स्कूल जाने वाले बच्चों को भी काफी मुसीबतों का सामना करना पड़ा 
     ''',height = 450)
    st.markdown("<h6 style='text-align: center;color: #808080;'>OR</h6>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # st.write(stringio)

        # To read file as string:
        string_data = stringio.read()
        # st.write(string_data)
        input_text = string_data
        # st.write(input_text)
        # https://discuss.streamlit.io/t/how-to-upload-a-pdf-file-in-streamlit/2428/3
    st.write('')
    
    generate_summary = st.button('Generate Summary')

summary ='''  दिल्ली के कई इलाकों में तेज बारिश से पानी जमा हो गया है। एयरपोर्ट के टर्मिनल पर पानी भर गया है। हालांकि विमानों की आवाजाही पर असर नहीं पड़ा है। '''
with col4:
    placeholder = st.empty()
    with placeholder.container():
        summarized_text = st.text_area('Summarized Text:', value = summary, height = 450)
        st.markdown('')
        download_btn = st.download_button(label='Download summary', data = summarized_text, file_name = 'Summary.txt')


if generate_summary:
    if model == 'LSTM':
        model = keras.models.load_model("saved_model")
        decoder_model = keras.models.load_model("decoder_saved_model")
        encoder_model = keras.models.load_model("encoder_saved_model")

        with open('x_tokenizer.json') as f:
            data = json.load(f)
            x_tokenizer = tokenizer_from_json(data)

        reverse_target_word_index = np.load('reverse_target_word_index.npy',allow_pickle='TRUE').item()
        target_word_index = np.load('target_word_index.npy',allow_pickle='TRUE').item()
        reverse_source_word_index = np.load('reverse_source_word_index.npy',allow_pickle='TRUE').item()
        
        max_text_len = 300
        max_summary_len = 40

        def preprocess_tokenize(text):
            text = str(text)
            text = re.sub(r'(\d+)', r'', text)
            text = text.replace('\n', '')
            text = text.replace('\r', '')
            text = text.replace('\t', '')
            text = text.replace('\u200d', '')
            text=re.sub("(__+)", ' ', str(text)).lower()   
            text=re.sub("(--+)", ' ', str(text)).lower()   
            text=re.sub("(~~+)", ' ', str(text)).lower()   
            text=re.sub("(\+\++)", ' ', str(text)).lower() 
            text=re.sub("(\.\.+)", ' ', str(text)).lower() 
            text=re.sub(r"[<>()|&©@#ø\[\]\'\",;:?.~*!]", ' ', str(text)).lower()
            text = re.sub("([a-zA-Z])",' ',str(text)).lower()
            text = re.sub("(\s+)",' ',str(text)).lower()
            return text

        

        def seq2text(input_seq):
            newString=''
            for i in input_seq:
                if(i!=0):
                    newString=newString+reverse_source_word_index[i]+' '
            return newString

        def decode_sequence(input_seq):
            e_out, e_h, e_c = encoder_model.predict(input_seq)
            target_seq = np.zeros((1,1))
            target_seq[0, 0] = target_word_index['sostok']
            stop_condition = False
            decoded_sentence = ''
            while not stop_condition:
                output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_token = reverse_target_word_index[sampled_token_index]
                if(sampled_token!='eostok'):
                    decoded_sentence += ' '+sampled_token
                if (sampled_token == 'eostok'  or len(decoded_sentence.split()) >= (max_summary_len-1)):
                    stop_condition = True
                target_seq = np.zeros((1,1))
                target_seq[0, 0] = sampled_token_index
                e_h, e_c = h, c
            return decoded_sentence

        text = input_text
        text = preprocess_tokenize(text)
        input = [text]

        ip_test_seq = x_tokenizer.texts_to_sequences(input)
        ip_test = pad_sequences(ip_test_seq, maxlen=max_text_len, padding='post')

        review = seq2text(ip_test[0]) 
        predicted_summary = decode_sequence(ip_test[0].reshape(1, max_text_len)) 
        summary = predicted_summary
        placeholder.empty()
        with col4:
            placeholder = st.empty()
            with placeholder.container():
                summarized_text = st.text_area('Summarized Text:', value = summary, height = 450)
                st.markdown('')
                download_btn = st.download_button(label='Download summary', data = summarized_text, file_name = 'Summary.txt')

    if model == 'PEGASUS':
        model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum") #.to('cuda')
        tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
        text = input_text

        max_length = max_words
        translator = Translator()
        text = translator.translate(text, src = 'hi', dest='en').text
        
        batch = tokenizer(text, truncation=True, padding='longest',max_length=1024, return_tensors="pt")#.to('cuda')
        tokens = model.generate(**batch,min_length=30, max_length = max_words) #changed min_length to max_length
        words = tokenizer.batch_decode(tokens,skip_special_tokens=True)[0]
        
        output = translator.translate(words, src = 'en', dest='hi').text
        summary = output

        placeholder.empty()
        with col4:
            placeholder = st.empty()
            with placeholder.container():
                summarized_text = st.text_area('Summarized Text:', value = summary, height = 450)
                st.markdown('')
                download_btn = st.download_button(label='Download summary', data = summarized_text, file_name = 'Summary.txt')
    
    if model == 'T5':
        model = T5ForConditionalGeneration.from_pretrained('t5-large')
        tokenizer = T5Tokenizer.from_pretrained('t5-large')
        text = input_text

        max_length = max_words
        translator = Translator()
        text = translator.translate(text, src = 'hi', dest='en').text

        preprocess_text = text.strip().replace("\n","")
        t5_prepared_Text = "summarize: "+preprocess_text

        tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt")

        summary_ids = model.generate(tokenized_text,
                                            num_beams=4,
                                            no_repeat_ngram_size=2,
                                            min_length=30,
                                            max_length=max_length,
                                            early_stopping=True)

        words = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        output = translator.translate(words, src = 'en', dest='hi').text
        summary = output

        placeholder.empty()
        with col4:
            placeholder = st.empty()
            with placeholder.container():
                summarized_text = st.text_area('Summarized Text:', value = summary, height = 450)
                st.markdown('')
                download_btn = st.download_button(label='Download summary', data = summarized_text, file_name = 'Summary.txt')
    
    if model == 'BART':
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

        text = input_text

        max_length = max_words
        translator = Translator()
        text = translator.translate(text, src = 'hi', dest='en').text

        inputs = tokenizer([text], max_length=1024, return_tensors="pt")

        # Generate Summary
        summary_ids = model.generate(inputs["input_ids"], num_beams=2, max_length=max_length)
        words = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        output = translator.translate(words, src = 'en', dest='hi').text
        summary = output

        placeholder.empty()
        with col4:
            placeholder = st.empty()
            with placeholder.container():
                summarized_text = st.text_area('Summarized Text:', value = summary, height = 450)
                st.markdown('')
                download_btn = st.download_button(label='Download summary', data = summarized_text, file_name = 'Summary.txt')