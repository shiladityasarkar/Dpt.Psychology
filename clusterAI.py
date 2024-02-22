import subprocess
from pathlib import Path
import pandas as pd        
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from transformers import BertModel, BertTokenizer
from sklearn.cluster import KMeans
import numpy as np
import torch
import re

class Generate:

    def __init__(shila, loc):
        # shila.requirements()
        shila.set_path()
        shila.model = None
        shila.tokenizer = None
        shila.enc = None
        shila.loc = loc
        try:
            shila.df = pd.read_excel(loc)
        except:
            try:
                shila.df = pd.read_csv(loc)
            except:
                raise AssertionError('invalid file type')
        # shila.download()
        shila.workflow()

    def prep(shila,text,just_filter:bool=False):
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        if just_filter:
            return filtered_tokens
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        return ' '.join(lemmatized_tokens)
        
    def wcloud(shila,name=None, df=None, column_name=None, li=None):
        if df is None and column_name==None and li==None:
            raise AssertionError('no parameters specified')
        elif li==None:
            if df is None or column_name==None:
                raise AssertionError('must specify df and column_name when li is None.')
        elif not df is None or column_name!=None:
            raise AssertionError('cannot specify df or column_name when li is not None.')
        if li!=None:
            prop = shila.prep(' '.join(li))
        else:
            hey = df[column_name].apply(shila.prep)
            prop = ' '.join(hey.astype(str).tolist())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(prop)
        wordcloud.to_file('wordclouds/'+name)

    def model_ini(shila,model_name):
        shila.model = BertModel.from_pretrained(model_name)
        shila.tokenizer = BertTokenizer.from_pretrained(model_name)

    def do_enc(shila,co):
        encoded_essays = []
        for essay in shila.df[co]:
            inputs = shila.tokenizer(essay, return_tensors="pt",
            padding=True, truncation=True)
            with torch.no_grad():
                output = shila.model(**inputs)
            encoded_essay = output.last_hidden_state.mean(dim=1).squeeze().numpy()
            encoded_essays.append(encoded_essay)
            shila.enc = np.array(encoded_essays)
    
    def posn(shila, text):
        nw = set(nltk.corpus.opinion_lexicon.negative())
        pw = set(nltk.corpus.opinion_lexicon.positive())
        filtered_tokens = shila.prep(text, just_filter=True)
        pos, neg = [],[]
        for token in filtered_tokens:
            if token.lower() in nw:
                neg.append(token)
            elif token.lower() in pw:
                pos.append(token)
        return pos, neg
    
    def cluster(shila, n, co):
        num_clusters = n
        kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
        cluster_labels = kmeans.fit_predict(shila.enc)
        match n:

            case 3:
                g1 = [essay for i, essay in enumerate(shila.df[co]) if cluster_labels[i] == 0]
                shila.wcloud(name='cluster3/c3_1.png',li=g1)
                g2 = [essay for i, essay in enumerate(shila.df[co]) if cluster_labels[i] == 1]
                shila.wcloud(name='cluster3/c3_2.png',li=g2)
                g3 = [essay for i, essay in enumerate(shila.df[co]) if cluster_labels[i] == 2]
                shila.wcloud(name='cluster3/c3_3.png',li=g3)

                shila.filing(g1, 'cluster3/c3_1.txt')
                shila.filing(g2, 'cluster3/c3_2.txt')
                shila.filing(g3, 'cluster3/c3_3.txt')

            case 4:
                g1 = [essay for i, essay in enumerate(shila.df[co]) if cluster_labels[i] == 0]
                shila.wcloud(name='cluster4/c4_1.png',li=g1)
                g2 = [essay for i, essay in enumerate(shila.df[co]) if cluster_labels[i] == 1]
                shila.wcloud(name='cluster4/c4_2.png',li=g2)
                g3 = [essay for i, essay in enumerate(shila.df[co]) if cluster_labels[i] == 2]
                shila.wcloud(name='cluster4/c4_3.png',li=g3)
                g4 = [essay for i, essay in enumerate(shila.df[co]) if cluster_labels[i] == 3]
                shila.wcloud(name='cluster4/c4_4.png',li=g4)

                shila.filing(g1, 'cluster4/c4_1.txt')
                shila.filing(g2, 'cluster4/c4_2.txt')
                shila.filing(g3, 'cluster4/c4_3.txt')
                shila.filing(g4, 'cluster4/c4_4.txt')

            case 5:
                g1 = [essay for i, essay in enumerate(shila.df[co]) if cluster_labels[i] == 0]
                shila.wcloud(name='cluster5/c5_1.png',li=g1)
                g2 = [essay for i, essay in enumerate(shila.df[co]) if cluster_labels[i] == 1]
                shila.wcloud(name='cluster5/c5_2.png',li=g2)
                g3 = [essay for i, essay in enumerate(shila.df[co]) if cluster_labels[i] == 2]
                shila.wcloud(name='cluster5/c5_3.png',li=g3)
                g4 = [essay for i, essay in enumerate(shila.df[co]) if cluster_labels[i] == 3]
                shila.wcloud(name='cluster5/c5_4.png',li=g4)
                g5 = [essay for i, essay in enumerate(shila.df[co]) if cluster_labels[i] == 3]
                shila.wcloud(name='cluster5/c5_6.png',li=g5)

                shila.filing(g1, 'cluster5/c5_1.txt')
                shila.filing(g2, 'cluster5/c5_2.txt')
                shila.filing(g3, 'cluster5/c5_3.txt')
                shila.filing(g4, 'cluster5/c5_4.txt')
                shila.filing(g5, 'cluster5/c5_5.txt')
            
            case _:
                raise Exception('an unknown error has occurred.')

    def filing(shila, essays, path):
        with open('clustered_responses/'+path, 'w') as file:
            file.write('Metadata: this file contains '+str(len(essays))+' responses.\n')
            file.write('===========================================\n\n')
            i = 0
            for essay in essays:
                i += 1
                match = -1
                for _ in shila.df['Person']:
                    if shila.df.iloc[_-1]['Text'] == essay:
                        match = _
                        break
                file.write('response of candidate : '+str(match)+'\n~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
                file.write(str(i)+'. '+ essay + '\n')
                pos, neg = shila.posn(essay)
                file.write('Post processing::\nPositive words are: ')
                for p in set(pos):
                    file.write(p+' ')
                file.write('\nNegative words are: ')
                for n in set(neg):
                    file.write(n+' ')
                file.write('\n------------------------------------------------------------------------------------------------------------\n------------------------------------------------------------------------------------------------------------\n\n')
        file.close()

    def download(shila):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('stopwords')
        nltk.download('opinion_lexicon')
        
    def workflow(shila):
        c = 'Text'
        shila.df.rename(columns={'S.No':'Person'}, inplace=True)
        shila.wcloud('initial.png',shila.df,'Text')
        shila.model_ini('bert-base-uncased')
        shila.do_enc(c)
        shila.cluster(3,c)
        shila.cluster(4,c)
        shila.cluster(5,c)
    
    def set_path(shila):
        Path("wordclouds/cluster3").mkdir(parents=True, exist_ok=True)
        Path("wordclouds/cluster4").mkdir(parents=True, exist_ok=True)
        Path("wordclouds/cluster5").mkdir(parents=True, exist_ok=True)
        Path("clustered_responses/cluster3").mkdir(parents=True, exist_ok=True)
        Path("clustered_responses/cluster4").mkdir(parents=True, exist_ok=True)
        Path("clustered_responses/cluster5").mkdir(parents=True, exist_ok=True)

    def requirements(shila):
        libs = ['nltk',
                'pandas',
                'scikit-learn',
                'transformers',
                'wordcloud',
                'numpy',
                'torch',
                'openpyxl']
        for lib in libs:
            subprocess.call(['pip','install',lib])