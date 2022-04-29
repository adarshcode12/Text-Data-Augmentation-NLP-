#run ---> streamlit run c:\pc\Desktop\impo.py  
import re
import streamlit as st

#NLTK Packages
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from nltk.stem import PorterStemmer
#SPACY Packages
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sentence_splitter import SentenceSplitter, split_text_into_sentences





def paraphrasing( paragraph):
    model_name = 'tuner007/pegasus_paraphrase'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    def get_response(input_text,num_return_sequences):
        batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
        translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text
    
    # Takes the input paragraph and splits it into a list of sentences
    splitter = SentenceSplitter(language='en')
    sentence_list = splitter.split(paragraph)
    paraphrase = []
    for i in sentence_list:
        a = get_response(i,1)
        paraphrase.append(a)
    paraphrase2 = [' '.join(x) for x in paraphrase]
    paraphrase3 = [' '.join(x for x in paraphrase2) ]
    final_text = str(paraphrase3).strip('[]').strip("'")
    return final_text




#Function for NLTK
def nltk_summarizer(docx):
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(docx)
    freqTable = dict()

    for word in words:
        word = word.lower()
        if word not in stopWords:
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1

    sentence_list= sent_tokenize(docx)
    #sentenceValue = dict()
    max_freq = max(freqTable.values())
    for word in freqTable.keys():
        freqTable[word] = (freqTable[word]/max_freq)

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in freqTable.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = freqTable[word]
                    else:
                        sentence_scores[sent] += freqTable[word]#total number of length of words

    import heapq
    summary_sentences = heapq.nlargest(8, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary

#Function for SPACY
def spacy_summarizer(docx):
    #nlp=spacy.load('en_core_web_lg')
    #docx=nlp(docx)
    stopWords = list(STOP_WORDS)
    words = word_tokenize(docx)
    freqTable = dict()

    for word in words:
        word = word.lower()
        if word not in stopWords:
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1

    sentence_list= sent_tokenize(docx)
    #sentenceValue = dict()
    max_freq = max(freqTable.values())
    for word in freqTable.keys():
        freqTable[word] = (freqTable[word]/max_freq)

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in freqTable.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = freqTable[word]
                    else:
                        sentence_scores[sent] += freqTable[word]#total number of length of words

    import heapq
    summary_sentences = heapq.nlargest(8, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary

def similarity(doc1,doc2):
    X_list = word_tokenize(doc1) 
    Y_list = word_tokenize(doc2)
    sw = stopwords.words('english') 

    # remove stop words from the string
    X_set = {w for w in X_list if not w in sw} 
    Y_set = {w for w in Y_list if not w in sw}


    ps = PorterStemmer()

    l1 =[];l2 =[]
    porter_stemmer1 = []
    for w in X_set:
        x = ps.stem(w)
        porter_stemmer1.append(x)
    
    porter_stemmer2 = []
    for w in Y_set:
        x = ps.stem(w)
        porter_stemmer2.append(x)
  
    
    a_set = set(porter_stemmer1)
    b_set= set(porter_stemmer2)
    rvector = a_set.union(b_set) 
    for w in rvector:
        if w in a_set: l1.append(1) # create a vector
        else: l1.append(0)
        if w in b_set: l2.append(1)
        else: l2.append(0)
    c = 0
    
    # cosine formula 
    for i in range(len(rvector)):
            c+= l1[i]*l2[i]
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    return cosine




def main():
    st.title("Text Data Augmentation")
    activities = ["Summarize Via Text","Paraphrase Text","Paraphrase Text With Summarization"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'Summarize Via Text':
        st.subheader("Summary using NLP")
        article_text = st.text_area("Enter Text Here")
        #cleaning of input text
        article_text = re.sub(r'\\[[0-9]*\\]', ' ',article_text)
        article_text = re.sub('[^a-zA-Z.,]', ' ',article_text)
        article_text = re.sub(r"\b[a-zA-Z]\b",'',article_text)
        article_text = re.sub("[A-Z]\Z",'',article_text)
        article_text = re.sub(r'\s+', ' ', article_text)

        summary_choice = st.selectbox("Summary Choice" , ["NLTK","SPACY"])
        if st.button("Summarize Via Text"):
            if summary_choice == 'NLTK':
                summary_result = nltk_summarizer(article_text)
            elif summary_choice == 'SPACY':
                summary_result = spacy_summarizer(article_text)

            st.write("SUMMARY OF THE GIVEN SENTENCE")
            st.write(summary_result)

    if choice == "Paraphrase Text":
        st.subheader("Paraphrase")
        article_text = st.text_area("Enter Text Here")
        #cleaning of input text
        article_text = re.sub(r'\\[[0-9]*\\]', ' ',article_text)
        article_text = re.sub('[^a-zA-Z.,]', ' ',article_text)
        article_text = re.sub(r"\b[a-zA-Z]\b",'',article_text)
        article_text = re.sub("[A-Z]\Z",'',article_text)
        article_text = re.sub(r'\s+', ' ', article_text)

        summary_choice = st.selectbox("Paraphrase Choice" , ["seq2seq"])
        if st.button("Paraphrase Text"):
            paraphrase_result=paraphrasing(article_text)
            st.write("PARAPHRASE OF THE GIVEN SENTENCE")
            st.write(paraphrase_result)
            st.write("SIMILARITY BETWEEN ORIGINAL SENTENCE AND PARAPHRASED SENTENCE")
            st.write(similarity(article_text,paraphrase_result))
    
    if choice == "Paraphrase Text With Summarization":
        st.subheader("Paraphrase With Summarization")
        article_text = st.text_area("Enter Text Here")
        #cleaning of input text
        article_text = re.sub(r'\\[[0-9]*\\]', ' ',article_text)
        article_text = re.sub('[^a-zA-Z.,]', ' ',article_text)
        article_text = re.sub(r"\b[a-zA-Z]\b",'',article_text)
        article_text = re.sub("[A-Z]\Z",'',article_text)
        article_text = re.sub(r'\s+', ' ', article_text)

        summary_choice = st.selectbox("Paraphrasing Choice" , ["seq2seq"])
        summary_choice2 = st.selectbox("Summary Choice" , ["NLTK","SPACY"])
        if st.button("Paraphrase + Summarization"):
            if summary_choice2 == 'NLTK':
                paraphrase_result = paraphrasing(article_text)
                original_summary = nltk_summarizer(article_text)
                summary_result=nltk_summarizer(paraphrase_result)
                st.write("PARAPHRASE OF THE GIVEN SENTENCE")
                st.write(paraphrase_result)
                st.write("SUMMARY OF THE GIVEN SENTENCE")
                st.write(original_summary)
                st.write("SUMMARY OF THE PARAPHRASED SENTENCE")
                st.write(summary_result)
                st.write("SIMILARITY BETWEEN SUMMARY OF ORIGINAL SENTENCE AND SUMMARY OF PARAPHRASE SENTENCE")
                st.write(similarity(original_summary,summary_result))
            if summary_choice2 == 'SPACY':
                paraphrase_result = paraphrasing(article_text)
                original_summary = spacy_summarizer(article_text)
                summary_result=spacy_summarizer(paraphrase_result)
                st.write("PARAPHRASE OF THE GIVEN SENTENCE")
                st.write(paraphrase_result)
                st.write("SUMMARY OF THE GIVEN SENTENCE")
                st.write(original_summary)
                st.write("SUMMARY OF THE PARAPHRASED SENTENCE")
                st.write(summary_result)
                st.write("SIMILARITY BETWEEN SUMMARY OF ORIGINAL SENTENCE AND SUMMARY OF PARAPHRASE SENTENCE")
                st.write(similarity(original_summary,summary_result))
       


if __name__=='__main__':
	main()