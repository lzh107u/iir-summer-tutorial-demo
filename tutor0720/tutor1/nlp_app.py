import os
import nltk
nltk.download( 'stopwords' )
nltk.download( 'punkt' )
from nltk.corpus import stopwords

from ast import literal_eval
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier

import pickle

REPLACE_BY_SPACE_RE = re.compile( '[/(){}\[\]\|@,;\?\-\'<>\*\:\`\.\"\=]' )
NUM_SYMBOL = re.compile( '[0-9]*[\.]?[0-9]+' )
CSHOP_REBUILD = re.compile('C#')
STOPWORDS = set(stopwords.words('english'))

REPLACE_RULES = [ REPLACE_BY_SPACE_RE, NUM_SYMBOL ]
CachedStopwords = stopwords.words( "english" )

def custom_token( text ):
    global CachedStopwords
    token_text = nltk.tokenize.word_tokenize( text )
    token_without_sw = []
    flag_stop = 0
    for index, seg in enumerate( token_text ):
        try:
            if token_text[ index ] == 'c' and token_text[ index + 1 ] == '#':
                token_text[ index ] = 'c#'
                seg = 'c#'
        except IndexError:
            # ignore the index error
            # theorically will occur when c is in the end of token
            pass
        
        if seg not in CachedStopwords:
            if flag_stop == 0:
                token_without_sw.append( seg )
                if seg == 'c#':
                    flag_stop = 1
            elif flag_stop == 1:
                flag_stop = 0
    
    # token_without_sw = [ word for word in token_text if not word in stopwords.words() ]
    
    return token_without_sw

def text_prepare( text ):
    """
        text: a string
        
        return: modified initial string
    """
    """
        execution cost examination in 2022.07.28:
        
        1. text = text.lower()
            => very low
        2. for loop of re.sub()
            => very low
        3. custom_token, word_tokenize()
            => very low
        4. custom_token, for loop of c# inspection
            => very low
        5. stopwords inspection
            => high, it's bottom-neck
            
        problem: 
        => Originally, it called stopwords.word() every time when insepcted each sentence,
            and it caused serious overhead during execution.
        => The problem can be solved by caching the stopword list, then it won't need to be called again.

    """
    global REPLACE_RULES
    token_without_sw = ['test']
    
    # print( '--------- round start ----------' )
    # print('input text:', text )
    # lowercase text
    text = text.lower()
    # print('lower case:', text )
    # replace REPLACE_BY_SPACE_RE symbols by space in text
    # print('---------- removing special symbols ...')
    for index, re_rule in enumerate( REPLACE_RULES ):
        text = re_rule.sub( '', text )
        # print( 'round', index, ':', text )
    
    # delete symbols which are in BAD_SYMBOLS_RE from text
    
    token_without_sw = custom_token( text )
    text = ' '.join( token_without_sw )
    return text, token_without_sw

def load_params():
    base_path = os.getcwd()
    print( 'load_params:', base_path )
    try: 
        with open( base_path + '/tutor1/nlp.pickle', 'rb' ) as f:
            if __name__ != '__main__':
                # nlp_params_dict = pickle.load( f )
                nlp_params_dict = {}
            else:
                nlp_params_dict = pickle.load( f )
    except FileNotFoundError :
        print( 'file not found.' )
        return None

    return nlp_params_dict

def main_task( sample_text = 'Symfony2 Forms - how to use dynamic select options' ):
    
    nlp_params = load_params()
    if nlp_params is None:
        return 'pickle file is not found.'
    print('main_task: finish loading')
    if __name__ == '__main__':
        vectorizer_tfidf = nlp_params[ 'tfidf_vectorizer' ]
        classifier_tfidf = nlp_params[ 'tfidf_classifier' ]
        label_binarizer = nlp_params[ 'binarizer' ]
    
        processed_text, text_token = text_prepare( sample_text )
        sample_input = [ processed_text ]
        sample_vector = vectorizer_tfidf.transform( sample_input )
        sample_predict = classifier_tfidf.predict( sample_vector )

        label_answer = label_binarizer.inverse_transform( sample_predict )
    else: 
        label_answer = 'sample label'
    
    return label_answer

if __name__ == '__main__':
    print( 'program start' )
    # nlp_params = load_params()
    label = main_task()
    print('nlp_app, main: label = ', label )
