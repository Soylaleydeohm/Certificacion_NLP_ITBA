# %%
# Código para el primer día de noticias
#######################
import sys

from bertopic import BERTopic
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from opensearch_data_model import Topic, TopicKeyword, TopicEntities, os_client, News, NewsEntities,NewsKeyword
from datetime import datetime
from dateutil.parser import parse
from utils import SPANISH_STOPWORDS
from collections import Counter
# %%
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from transformers import pipeline
# %%
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

from datasets import load_dataset

import functions
# %%


# %%        
def ver_histograma_threshold(info_por_topico, cant_topicos, docs_per_topic_prob):
    for topic in range(cant_topicos):
        if (topic > 0): #No guardo el -1
            topico = topic-1
            print(f'Tópico a determinar threshold :{info_por_topico[info_por_topico.Topic == topico].label.values[0]}')
            mean = np.mean(docs_per_topic_prob[topic]['prob_topic'])
            std = np.std(docs_per_topic_prob[topic]['prob_topic'])
            info_por_topico.loc[info_por_topico.Topic == topico,'threshold'] = mean
            info_por_topico.loc[info_por_topico.Topic == topico,'docs_sobre_threshold'] = docs_per_topic_prob[topic][docs_per_topic_prob[topic]['prob_topic'] >= mean].count().values[0]
            print(f'Promedio: {round(mean,2)} - Dev std: {round(std,2)}')
            plt.hist(docs_per_topic_prob[topic]['prob_topic'], bins =np.arange(0, 1.1, 0.1))
            plt.axvline(mean, color = 'pink', linestyle = 'dashed', label = 'Promedio')
            plt.axvline(mean - std, color = 'blue', linestyle = 'dashed', label = '-Desviación estándar')
            plt.title(f'Histograma del tópico {info_por_topico[info_por_topico.Topic == topico].label.values[0]}')
            plt.legend()
            plt.show()

# %%
def cargo_noticias_primer_dia(date,n_sampleo):
#Cargo la info de primer día
    path_file = f"jganzabalseenka/news_{date}_24hs"
    dataset = load_dataset(path_file)
    df_dataset = pd.DataFrame(dataset['train'])
    print(df_dataset.head(1))
    #Los documentos traídos no necesariamente corresponden al día que se indica
    fecha = "".join(date.split('-'))
    df_dataset.sort_values("start_time_local", ascending=True, inplace=True)
    df = df_dataset[df_dataset['start_time_local'].dt.date == pd.to_datetime(fecha).date()]
    print(f"Noticias de la fecha {date}: {len(df)} de {len(df_dataset)} = {round(100*len(df)/len(df_dataset),2)}%")
    #Sampleo
    df = df.sample(n=n_sampleo, random_state=1)
    return df

# %%
def main(args):
    if len(args) > 1:
        print(f"Comienza el análisis del día {args}")
        functions.borrar_base()
        n_sampleo = 1000
        functions.inicializar_base()
        df = cargo_noticias_primer_dia(args,n_sampleo)
        topic_model, data, index_docs, title_docs, entities_all = functions.modelo_topicos(df)
        functions.procesamiento(topic_model, data, df, index_docs, title_docs,n_sampleo,0, 0, args, entities_all)
    else:
        print("No se proporcionó ningún día.")

if __name__ == "__main__":
    main(sys.argv)