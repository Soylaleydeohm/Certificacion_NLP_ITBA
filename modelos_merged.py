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
# %%
Topic.init()
News.init()
# %%
#Listado de noticias guardadas
lista_noticias = []
for i, doc in enumerate(News.search().query().scan()):
    print(f"{doc.to_dict()['index']}: {doc.to_dict()['name']}")
    lista_noticias.append(doc.to_dict())
df_anterior = pd.DataFrame(lista_noticias)
# %%
ultima_fecha_guardada = df_anterior.date.max().date()
ultima_fecha_guardada
# %%
ultima_noticia_index = df_anterior.index.max()
ultima_noticia_index
# %%
#Tópicos guardados previamente
ultimo_topico_guardado = Topic.search().count()
ultimo_topico_guardado
# %%
#cargamos datasets de noticias a incorporar
fechas = ['2024-07-11'] # ['2024-07-09']
df_dataset = {}
fechas_a_incorporar = 0
# %%
for i in range(len(fechas)):
    date = fechas[i]
    print(f'Se cargan noticias nuevas de la fecha {date}')
    path_file = f"jganzabalseenka/news_{date}_24hs"
    dataset = load_dataset(path_file)
    df_dataset_i = pd.DataFrame(dataset['train'])
    #Los documentos traídos no necesariamente corresponden al día que se indica
    fecha = "".join(date.split('-'))
    df_dataset_i.sort_values("start_time_local", ascending=True, inplace=True)
    df_dataset[fechas_a_incorporar] = df_dataset_i[df_dataset_i['start_time_local'].dt.date == pd.to_datetime(fecha).date()]
    print(f"Noticias de la fecha {date}: {len(df_dataset[fechas_a_incorporar])} de {len(df_dataset_i)} = {round(100*len(df_dataset[fechas_a_incorporar])/len(df_dataset_i),2)}%")
    #Sampleo
    n_sampleo = 1000
    df_dataset[fechas_a_incorporar] = df_dataset[fechas_a_incorporar].sample(n=n_sampleo, random_state=1)
    fechas_a_incorporar += 1
# %%
#dataset completo
df_a_analizar = pd.concat([df_dataset[i] for i in range(fechas_a_incorporar)], ignore_index = True)
# %%
data = list(df_a_analizar['text'])
index_docs = list(df_a_analizar['asset_id'])
title_docs = list(df_a_analizar['title'])
# %%
# Calculamos embeddings de las noticias
topic_model = BERTopic.load(f"modelo_topicos_actualizado")
new_noticias_embed = topic_model.embedding_model.embed(data)#matchea con alguno de los tópicos previos?
new_noticias_embed
# %%
#Búsqueda en la base
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
# %%
#Por noticia
winning_topic = {}
probs = []
topics = []
name_topic = []
index_best_doc = []
threshold = []
no_encontro = 0
for i in range(n_sampleo):
    print(df['title'].iloc[i])    
    if(topico_asignado[i]['hits']['total']['value'] > 0):
        df_hits = pd.DataFrame(topico_asignado[i]['hits']['hits'][0])
        winning_topic[i] = Topic.get(df_hits.iloc[0]._id)
        probs.append(topico_asignado[i]['hits']['max_score'] )
        topics.append(topico_asignado[i]['hits']['hits'][0]['_source']['index'])
        name_topic.append(topico_asignado[i]['hits']['hits'][0]['_id'])
        index_best_doc.append(topico_asignado[i]['hits']['hits'][0]['_source']['index_best_doc'])
        threshold.append(topico_asignado[i]['hits']['hits'][0]['_source']['similarity_threshold'])
        display(winning_topic[i].to_dict())
    else:
        print('No se encontró el tópico al que corresponde el texto')
        no_encontro += 1
        probs.append(0)
        topics.append(-1)
        name_topic.append(-1)
        index_best_doc.append(-1)
        threshold.append(0)
print(f'No encontró tópicos en la base para {no_encontro} documentos')
# %%
#Hay que asignar las probabilidades y tópicos a las noticias nuevas. 
df_a_analizar['topic'] = topics
df_a_analizar['name_topic'] = name_topic
df_a_analizar['threshold_esperado'] = threshold
df_a_analizar['probs'] = probs #Similitud por documento a cada tópico si calculate_probabilities=True, sino una sola prob.
df_a_analizar[['asset_id','title','topic','probs','name_topic']]
# %%
#Calculo sentiment en noticias nuevas
#trabajo con el título de las noticias por el tamaño
classifier = pipeline('sentiment-analysis',model="nlptown/bert-base-multilingual-uncased-sentiment")
outputs = classifier(title_docs)
# %%
#Guardo sentiment en el set de datos
df_a_analizar['sentiment'] = [list(outputs[i].values())[0] for i in range(len(title_docs))]
df_a_analizar['score'] = [list(outputs[i].values())[1] for i in range(len(title_docs))]
# %%
#########################################################################
#Analicemos las noticias que no estan asociadas a un tópico previo
df_sin_topicos = df_a_analizar[df_a_analizar['threshold_esperado'] > df_a_analizar['probs']]
df_sin_topicos
# %%
if(no_encontro > 0):
    print('Se incorporan documentos con tópico -1')
    df_sin_topicos = pd.concat([df_sin_topicos, df_a_analizar[df_a_analizar['topic'] == -1]])
df_sin_topicos
# %%
################################################################################
# df con noticias y tópicos listos
df_a_analizar['embedding'] = [new_noticias_embed[i] for i in range(len(title_docs))]
# %%
df_procesado = df_a_analizar[~df_a_analizar.asset_id.isin(df_sin_topicos.asset_id)]
df_procesado
# %%
#Listado de tópicos guardados
df_topicos_base = pd.DataFrame(columns = ['vector', 'similarity_threshold', 'created_at', 'to_date', 'from_date', 'index', 'total_docs', 'keywords', 'entities', 'name', 'best_doc', 'index_best_doc', 'title_best_doc'])
for i, doc in enumerate(Topic.search().query().scan()):
    df_topicos_base.loc[i] = doc.to_dict()
df_topicos_base
# %%
#Recálculo del threshold
docs_per_topic_prob = {}
for topic in range(ultimo_topico_guardado):
    print(f'Tópico :{df_topicos_base[df_topicos_base.index == topic].name.values[0]}')
    docs_per_topic_prob[topic] = df_procesado[df_procesado['topic'] == topic][['asset_id','title','topic','probs','start_time_local']]
    docs_per_topic_prob[topic].columns = ['id','title_doc','topic','prob_topic','ultima_fecha']
    docs_per_topic_prob[topic].sort_values(['prob_topic'], inplace = True, ascending = False)
    print(docs_per_topic_prob[topic])
docs_per_topic_prob
#%%
#Sólo cálculos para threshold
for topic in range(ultimo_topico_guardado):
    mean = (docs_per_topic_prob[topic]['prob_topic'].sum() + df_topicos_base.iloc[topic]['similarity_threshold']*df_topicos_base.iloc[topic]['total_docs'])/(len(docs_per_topic_prob[topic])+df_topicos_base.iloc[topic]['total_docs'])
    info_por_topico.loc[info_por_topico.Topic == topic,'threshold'] = mean
    info_por_topico.loc[info_por_topico.Topic == topic,'Count'] = docs_per_topic_prob[topic][docs_per_topic_prob[topic]['prob_topic'] >= mean].count().values[0] + df_topicos_base.iloc[topic]['total_docs']
    if (len(docs_per_topic_prob[topic]) >0):
        info_por_topico.loc[info_por_topico.Topic == topic,'ultima_fecha'] = docs_per_topic_prob[topic]['ultima_fecha'].max()
info_por_topico
# %%
# #Borramos los tópicos de la base para poder actualizar
response = os_client.indices.delete(
    index = 'topicos'
)
print(response)
#%%
def get_topic_keywords(topic):
    try: 
        if(len(df_topicos_base['keywords'][topic]) > 0):
            topic_keywords = df_topicos_base['keywords'][topic]
    except: 
        topic_keywords = []
    return topic_keywords
#%%
def get_topic_entities(topic):
    try:
        if(len(df_topicos_base['entities'][topic]) > 0):
            topic_entities = df_topicos_base['entities'][topic]
    except: 
        topic_entities = []
    return topic_entities
#%%
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
            keywords = get_topic_keywords(topic),
            entities = get_topic_entities(topic),
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
            keywords = get_topic_keywords(topic),
            entities = get_topic_entities(topic),
            name = df_topicos_base['name'][topic],
            best_doc = df_topicos_base['best_doc'][topic],
            index_best_doc = df_topicos_base['index_best_doc'][topic],
            title_best_doc = df_topicos_base['title_best_doc'][topic],
            )

    print(topic_doc.save())
# %%
################################################################################
############## Modelo para nuevo dataset de datos ##############
entities_all = set(sum(list([list(e) for e in df_sin_topicos['entities'].values]), []))
keywords_all = set(sum(list([list(e) for e in df_sin_topicos['keywords'].values]), []))
all_tokens = list(entities_all.union(keywords_all))
# %%
tf_vectorizer = CountVectorizer(
    # tokenizer=tokenizer,
    # max_df=0.1,
    # min_df=10,
    ngram_range=(1, 3),
    stop_words=SPANISH_STOPWORDS,
    lowercase=False,
    vocabulary=all_tokens,
    # max_features=100_000
)
tf_vectorizer.fit(data)
# %%
# Step 1 - Extract embeddings
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# %%
# Step 2 - Reduce dimensionality
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
# %%
# Step 3 - Cluster reduced embeddings
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
# %%
# Step 4 - Tokenize topics
vectorizer_model = tf_vectorizer
# %%
# Step 5 - Create topic representation
ctfidf_model = ClassTfidfTransformer()
# %%
# Step 6 - (Optional) Fine-tune topic representations with 
# a `bertopic.representation` model
representation_model = KeyBERTInspired()
# %%
topic_model_new = BERTopic(
    embedding_model=embedding_model,          # Step 1 - Extract embeddings
    umap_model=umap_model,                    # Step 2 - Reduce dimensionality
    hdbscan_model=hdbscan_model,              # Step 3 - Cluster reduced embeddings
    vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
    ctfidf_model=ctfidf_model,                # Step 5 - Extract topic words
    representation_model=representation_model, # Step 6 - (Optional) Fine-tune topic represenations
    language='spanish',
    #calculate_probabilities=True, #Quiero detectar el principal tópico al que pertenece el documento.
)
# %%
topics, probs = topic_model_new.fit_transform(list(df_sin_topicos['text']))
# %%
#Embeddings
embeddings_docs_new = topic_model_new.embedding_model.embed(list(df_sin_topicos['text']))
# %%
cant_topicos = len(Counter(topics).values())
cant_topicos
# %%
#Casos por tópico
info_por_topico_new = topic_model_new.get_topic_freq().sort_values(['Topic'])
info_por_topico_new['label'] = topic_model_new.generate_topic_labels()
info_por_topico_new.sort_values(['Count'], inplace = True, ascending = False)
info_por_topico_new
# %%
df_sin_topicos['topic'] = topics
df_sin_topicos['probs'] = probs
df_sin_topicos['name_topic'] = info_por_topico_new.loc[info_por_topico_new['Topic'] == df_sin_topicos['topic'].values[0],'label'].values[0]
df_sin_topicos['embedding'] = embeddings_docs_new.tolist()
# %%
df_procesado = pd.concat([df_procesado,df_sin_topicos])
# %%
# Guardamos noticias nuevas en la base
for news in range(n_sampleo):
        #print(df_procesado['title'].values[news])
        news_doc = News(
            index = news + ultima_noticia_index,
            vector = df_procesado['embedding'].values[news],#embedding
            score = df_procesado['probs'].values[news], #prob del tópico
            date = pd.to_datetime(df_procesado['start_time_local'].values[news]), #fecha de la noticia
            topico = df_procesado['topic'].values[news], #nro tópico al que pertenece
            name_topic = df_procesado['name_topic'].values[news], #nombre tópico al que pertenece
            sentiment = df_procesado['sentiment'].values[news], #nombre sentiment al que pertenece
            score_sentiment =  df_procesado['score'].values[news], #prob del tópico
            keywords = [NewsKeyword(name=k) for k in df_procesado.iloc[news]['keywords']],#keywords de la noticia
            entities = [NewsEntities(name=k) for k in df_procesado.iloc[news]['entities']],#entities de la noticia
            text_doc = df_procesado['text'].values[news], #texto del documento 
            index_doc = df_procesado['asset_id'].values[news], #index del documento original
            name = df_procesado['title'].values[news], #título del documento
            sitio_web = df_procesado['Asset Destination'].values[news], #sitio del que se obtuvo el documento
            media = df_procesado['media'].values[news], #nombre del medio del que se obtuvo el documento
        )
        print(news_doc.save())
# %%
len(df_procesado['embedding'][0])
# %%
# Guardamos modelo mergeado para futuras noticias
topic_model_merged = BERTopic.merge_models([topic_model, topic_model_new])
topic_model_merged.save(f"modelo_topicos_actualizado")
# %%
#Determinación del threshold para nuevos tópicos. Primero analizamos las probabilidades en cada documento
docs_per_topic_prob = {}
for topic in range(cant_topicos):
    topico = topic-1
    print(f'Tópico :{info_por_topico_new[info_por_topico_new.Topic == topico].label.values[0]}')
    docs_per_topic_prob[topic] = df_sin_topicos[df_sin_topicos['topic'] == topico][['asset_id','title','topic','probs']]
    docs_per_topic_prob[topic].columns = ['id','title_doc','topic','prob_topic']
    docs_per_topic_prob[topic].sort_values(['prob_topic'], inplace = True, ascending = False)
    print(docs_per_topic_prob[topic])
#%%
#Sólo cálculos para threshold
for topic in range(cant_topicos):
    if (topic > 0): #No guardo el -1
        topico = topic-1
        mean = np.mean(docs_per_topic_prob[topic]['prob_topic'])
        info_por_topico_new.loc[info_por_topico_new.Topic == topico,'threshold'] = mean
        info_por_topico_new.loc[info_por_topico_new.Topic == topico,'docs_sobre_threshold'] = docs_per_topic_prob[topic][docs_per_topic_prob[topic]['prob_topic'] >= mean].count().values[0]
#%%
#matriz similutd coseno
sim_matrix = cosine_similarity(
    topic_model_new.topic_embeddings_,
    embeddings_docs_new
)
# %%
def get_topic_name(keywords): #Nombro con los primeros keywords al tópico
    return ', '.join([k for k, s in keywords[:4]])
# %%
entities_all = set(sum(list([list(e) for e in df_a_analizar['entities'].values]), []))
# %%
def get_topic_keywords(topic):
    entities = set(list(zip(*topic_model_new.topic_representations_[topic]))[0]).intersection(set(entities_all))
    keywords_in_topic = [item for item in list(list(zip(*topic_model_new.topic_representations_[topic]))[0]) if item not in list(entities)]
    keywords_from_topic = [[item for item in topic_model_new.topic_representations_[topic] if list(keywords_in_topic)[i] in item] for i in range(len(keywords_in_topic))]
    topic_keywords = [TopicKeyword(name=k, score=s) for k, s in keywords_from_topic]
    return topic_keywords
# %%
def get_topic_entities(topic):
    entities = set(list(zip(*topic_model_new.topic_representations_[topic]))[0]).intersection(set(entities_all))
    entities_from_topic = [[item for item in topic_model_new.topic_representations_[topic] if list(entities)[i] in item] for i in range(len(entities))]
    topic_entities = [TopicEntities(name=k, score=s) for k, s in entities_from_topic]
    return topic_entities

# %%
for topic in topic_model_new.get_topics().keys():
    if topic > -1:
        topic_threshold = info_por_topico_new[info_por_topico_new.Topic == topic]['threshold'].values[0]

        best_doc_index = sim_matrix[topic + 1].argmax()
        title_best_doc = title_docs[best_doc_index]
        best_document = data[best_doc_index]        

        topic_doc = Topic(
            vector = list(topic_model.topic_embeddings_[topic + 1]),# vector del tópico, el 0 es -1
            similarity_threshold = topic_threshold, #Sale del análisis previo con histogramas
            created_at = datetime.now(),
            to_date = datetime.strptime(date, '%Y-%m-%d')+timedelta(1),#Agrego un día para que lo testee al siguiente
            from_date = parse(date),
            index = topic+ultimo_topico_guardado,
            total_docs = info_por_topico_new[info_por_topico_new.Topic == topic].Count.values[0],
            keywords = get_topic_keywords(topic),
            entities = get_topic_entities(topic),
            name = get_topic_name(topic_model_new.topic_representations_[topic]),
            best_doc = best_document,
            index_best_doc = best_doc_index,
            title_best_doc = title_best_doc,
        )

        print(topic_doc.save())
# %%
Topic.search().count()
# %%
lista_topicos = []
for i, doc in enumerate(Topic.search().query().scan()):
    print(f"{doc.to_dict()['index']}: {doc.to_dict()['name']}")
    print(f"Título mejor documento: {doc.to_dict()['title_best_doc']}")
    print(f"Threshold: {doc.to_dict()['similarity_threshold']}")
    lista_topicos.append(doc.to_dict())
# %%
len(lista_topicos)
