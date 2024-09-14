# %%
from opensearch_data_model import Topic, TopicKeyword,TopicEntities, os_client, News, NewsEntities,NewsKeyword
from bertopic import BERTopic
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from dateutil.parser import parse
from datetime import timedelta
from collections import Counter
# %%
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from transformers import pipeline
from utils import SPANISH_STOPWORDS
import functions
# %%


# %%
def updatear_base(ultimo_topico_guardado,df_topicos_base,info_por_topico,docs_per_topic_prob): 
    #Actualizamos sólo threshold, documentos totales y última fecha en que se usó el tópico
    for topic in range(ultimo_topico_guardado):
        print(topic) 
        if (len(docs_per_topic_prob[topic]) > 0):
            topic_doc = Topic(            
                vector = df_topicos_base['vector'][topic],
                similarity_threshold = info_por_topico.loc[info_por_topico.Topic == topic,'threshold'].values[0], 
                created_at = df_topicos_base['created_at'][topic],
                to_date = pd.to_datetime(info_por_topico.loc[info_por_topico.Topic == topic,'ultima_fecha'].values[0], '%Y-%m-%d')+timedelta(1),#Agrego un día para que lo testee al siguiente
                from_date = df_topicos_base['from_date'][topic],
                index = df_topicos_base['index'][topic],
                total_docs = info_por_topico.loc[info_por_topico.Topic == topic,'Count'].values[0],
                keywords = df_topicos_base['keywords'][topic],
                entities = df_topicos_base['entities'][topic],
                name = df_topicos_base['name'][topic],
                best_doc = df_topicos_base['best_doc'][topic],#Se debería modificar de ser necesario
                index_best_doc = df_topicos_base['index_best_doc'][topic],#Se debería modificar de ser necesario
                title_best_doc = df_topicos_base['title_best_doc'][topic],#Se debería modificar de ser necesario      
            )
        else:
            topic_doc = Topic(
                vector = df_topicos_base['vector'][topic],
                similarity_threshold = df_topicos_base['similarity_threshold'][topic], 
                created_at = df_topicos_base['created_at'][topic],
                to_date = df_topicos_base['to_date'][topic],
                from_date = df_topicos_base['from_date'][topic],
                index = df_topicos_base['index'][topic],
                total_docs = df_topicos_base['total_docs'][topic],
                keywords = df_topicos_base['keywords'][topic],
                entities = df_topicos_base['entities'][topic],
                name = df_topicos_base['name'][topic],
                best_doc = df_topicos_base['best_doc'][topic],
                index_best_doc = df_topicos_base['index_best_doc'][topic],
                title_best_doc = df_topicos_base['title_best_doc'][topic],
                )

        print(topic_doc.save())

# %%
def recalculo_threshold(ultimo_topico_guardado, df_topicos_base, df_procesado, info_por_topico):
    docs_per_topic_prob = {}
    for topic in range(ultimo_topico_guardado):
        print(f'Tópico :{df_topicos_base[df_topicos_base.index == topic].name.values[0]}')
        docs_per_topic_prob[topic] = df_procesado[df_procesado['topic'] == topic][['asset_id','title','topic','probs','start_time_local']]
        docs_per_topic_prob[topic].columns = ['id','title_doc','topic','prob_topic','ultima_fecha']
        docs_per_topic_prob[topic].sort_values(['prob_topic'], inplace = True, ascending = False)
        print(docs_per_topic_prob[topic])
    for topic in range(ultimo_topico_guardado):
        mean = (docs_per_topic_prob[topic]['prob_topic'].sum() + df_topicos_base.iloc[topic]['similarity_threshold']*df_topicos_base.iloc[topic]['total_docs'])/(len(docs_per_topic_prob[topic])+df_topicos_base.iloc[topic]['total_docs'])
        info_por_topico.loc[info_por_topico.Topic == topic,'threshold'] = mean
        info_por_topico.loc[info_por_topico.Topic == topic,'Count'] = docs_per_topic_prob[topic][docs_per_topic_prob[topic]['prob_topic'] >= mean].count().values[0] + df_topicos_base.iloc[topic]['total_docs']
        if (len(docs_per_topic_prob[topic]) >0):
            info_por_topico.loc[info_por_topico.Topic == topic,'ultima_fecha'] = docs_per_topic_prob[topic]['ultima_fecha'].max()
    return docs_per_topic_prob
# %%
def busqueda_en_base(new_noticias_embed,df_a_analizar,n_sampleo):
    topico_asignado = {}
    for i in range(n_sampleo):
        query = {#request
            "size": 1,
            "query": {
                "knn": {
                    "vector": {
                        "vector": list(new_noticias_embed[i]),
                        "k" : 5,
                    }
                }
            }
        }
        topico_asignado[i] = os_client.search(index='topicos', body=query)#que me traiga el más cercano
    #Por noticia
    winning_topic = {}
    probs = []
    topics = []
    name_topic = []
    index_best_doc = []
    threshold = []
    entities = []
    keywords = []
    no_encontro = 0
    for i in range(n_sampleo):
        print(df_a_analizar['title'].iloc[i])    
        if(topico_asignado[i]['hits']['total']['value'] > 0):
            df_hits = pd.DataFrame(topico_asignado[i]['hits']['hits'][0])
            winning_topic[i] = Topic.get(df_hits.iloc[0]._id)
            probs.append(topico_asignado[i]['hits']['max_score'] )
            topics.append(topico_asignado[i]['hits']['hits'][0]['_source']['index'])
            name_topic.append(topico_asignado[i]['hits']['hits'][0]['_id'])
            index_best_doc.append(topico_asignado[i]['hits']['hits'][0]['_source']['index_best_doc'])
            threshold.append(topico_asignado[i]['hits']['hits'][0]['_source']['similarity_threshold'])
            entities.append(topico_asignado[i]['hits']['hits'][0]['_source']['entities'])
            keywords.append(topico_asignado[i]['hits']['hits'][0]['_source']['keywords'])
            display(winning_topic[i].to_dict())
        else:
            print('No se encontró el tópico al que corresponde el texto')
            no_encontro += 1
            probs.append(0)
            topics.append(-1)
            name_topic.append(-1)
            index_best_doc.append(-1)
            threshold.append(0)
            entities.append('-')
            keywords.append('-')
    print(f'No encontró tópicos en la base para {no_encontro} documentos')
    #Hay que asignar las probabilidades y tópicos a las noticias nuevas. 
    df_a_analizar['topic'] = topics
    df_a_analizar['name_topic'] = name_topic
    df_a_analizar['threshold_esperado'] = threshold
    df_a_analizar['entities'] = entities
    df_a_analizar['keywords'] = keywords
    df_a_analizar['probs'] = probs #Similitud por documento a cada tópico si calculate_probabilities=True, sino una sola prob.
    return no_encontro
# %%
def procesamiento_base(df_a_analizar, ultimo_topico_guardado, ultima_noticia_index, date, n_sampleo):
    data = list(df_a_analizar['text'])
    index_docs = list(df_a_analizar['asset_id'])
    title_docs = list(df_a_analizar['title'])
    # Calculamos embeddings de las noticias
    topic_model = BERTopic.load(f"modelo_topicos_actualizado")
    new_noticias_embed = topic_model.embedding_model.embed(data)#matchea con alguno de los tópicos previos?
    no_encontro = busqueda_en_base(new_noticias_embed,df_a_analizar,n_sampleo)
    #Calculo sentiment en noticias nuevas
    #trabajo con el título de las noticias por el tamaño
    classifier = pipeline('sentiment-analysis',model="nlptown/bert-base-multilingual-uncased-sentiment")
    outputs = classifier(title_docs)
    #Guardo sentiment en el set de datos
    df_a_analizar['sentiment'] = [list(outputs[i].values())[0] for i in range(len(title_docs))]
    df_a_analizar['score'] = [list(outputs[i].values())[1] for i in range(len(title_docs))]
    #########################################################################
    #Analicemos las noticias que no estan asociadas a un tópico previo
    df_sin_topicos = df_a_analizar[df_a_analizar['threshold_esperado'] > df_a_analizar['probs']]
    # %%
    if(no_encontro > 0): #No debería entrar aquí porque no se guarda tópico -1
        print('Se incorporan documentos con tópico -1')
        df_sin_topicos = pd.concat([df_sin_topicos, df_a_analizar[df_a_analizar['topic'] == -1]])
    ################################################################################
    # df con noticias y tópicos listos
    df_a_analizar['embedding'] = [new_noticias_embed[i] for i in range(len(title_docs))]
    df_procesado = df_a_analizar[~df_a_analizar.asset_id.isin(df_sin_topicos.asset_id)]
    #Listado de tópicos guardados
    df_topicos_base = pd.DataFrame(columns = ['vector', 'similarity_threshold', 'created_at', 'to_date', 'from_date', 'index', 'total_docs', 'keywords', 'entities', 'name', 'best_doc', 'index_best_doc', 'title_best_doc'])
    for i, doc in enumerate(Topic.search().query().scan()):
        df_topicos_base.loc[i] = doc.to_dict()
    #Casos por tópico
    info_por_topico = topic_model.get_topic_freq().sort_values(['Topic'])
    info_por_topico['label'] = topic_model.generate_topic_labels()
    info_por_topico.sort_values(['Count'], inplace = True, ascending = False)

    docs_per_topic_prob = recalculo_threshold(ultimo_topico_guardado, df_topicos_base, df_procesado,info_por_topico)
    functions.borrar_base()
    updatear_base(ultimo_topico_guardado,df_topicos_base,info_por_topico,docs_per_topic_prob)  

    topic_model_new, data, index_docs, title_docs, entities_all = functions.modelo_topicos(df_sin_topicos)
    functions.procesamiento(topic_model_new, data, df, index_docs, title_docs,n_sampleo,ultimo_topico_guardado, ultima_noticia_index, date, entities_all)

# %%
def cargo_noticias_nuevas(date,n_sampleo):
    print(f'Se cargan noticias nuevas de la fecha {date}')
    path_file = f"jganzabalseenka/news_{date}_24hs"
    dataset = load_dataset(path_file)
    df_dataset_i = pd.DataFrame(dataset['train'])
    #Los documentos traídos no necesariamente corresponden al día que se indica
    fecha = "".join(date.split('-'))
    df_dataset_i.sort_values("start_time_local", ascending=True, inplace=True)
    df_dataset = df_dataset_i[df_dataset_i['start_time_local'].dt.date == pd.to_datetime(fecha).date()]
    print(f"Noticias de la fecha {date}: {len(df_dataset)} de {len(df_dataset_i)} = {round(100*len(df_dataset)/len(df_dataset_i),2)}%")
    #Sampleo
    n_sampleo = 1000
    df_dataset= df_dataset.sample(n=n_sampleo, random_state=1)
    return df_dataset
# %%
def carga_noticias_base():
    print('Noticias de días previos guardadas en la base.')
    lista_noticias = []
    for i, doc in enumerate(News.search().query().scan()):
        print(f"{doc.to_dict()['index']}: {doc.to_dict()['name']}")
        lista_noticias.append(doc.to_dict())
    df_anterior = pd.DataFrame(lista_noticias)
    return df_anterior
# %%
def main(args):
    if len(args) > 1:
        print(f"Comienza el análisis del día {args}")
        n_sampleo = 1000
        #Listado de noticias guardadas
        df_anterior = carga_noticias_base()
        ultima_fecha_guardada = df_anterior.date.max().date()
        ultima_noticia_index = df_anterior.index.max()
        #Tópicos guardados previamente
        ultimo_topico_guardado = Topic.search().count()
        df_a_analizar = cargo_noticias_nuevas(args,n_sampleo)
        procesamiento_base(df_a_analizar, ultimo_topico_guardado, ultima_noticia_index, args, n_sampleo)

    else:
        print("No se proporcionó ningún día.")

if __name__ == "__main__":
    main(sys.argv)
