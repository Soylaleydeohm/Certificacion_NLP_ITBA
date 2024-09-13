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
# %%
def buscar_noticia_base(new_doc):
    topic_model = BERTopic.load(f"modelo_topicos_actualizado")
    new_doc_embed = topic_model.embedding_model.embed(new_doc)#matchea con alguno de los tópicos previos?
    #Búsqueda en la base
    query = {#request
        "size": 1,
        "query": {
            "knn": {
                "vector": {
                    "vector": list(new_doc_embed),
                    "k" : 10,
                }
            }
        }
    }
    response = os_client.search(index='topicos', body=query)#que me traiga el más cercano
    if(response['hits']['total']['value'] > 0):
        df_hits = pd.DataFrame(response['hits']['hits'][0])
        winning_topic = Topic.get(df_hits.iloc[0]._id)
        display(winning_topic.to_dict())
    else:
        print('No se encontró el tópico al que corresponde el texto')
# %%
def ver_noticias():
    for i, doc in enumerate(News.search().query().scan()):
        print(f"{doc.to_dict()['index']}: {doc.to_dict()['name']}")
# %%    
def ver_topicos():
    for i, doc in enumerate(Topic.search().query().scan()):
        print(f"{doc.to_dict()['index']}: {doc.to_dict()['name']}")
        print(f"Título mejor documento: {doc.to_dict()['title_best_doc']}")
        print(f"Threshold: {doc.to_dict()['similarity_threshold']}")
# %% 
def guardar_noticias(df,info_por_topico,n_sampleo, inicio_noticias):
    for news in range(n_sampleo):
        print(df['title'].values[news])
        news_doc = News(
            index = news+inicio_noticias,
            vector = df['embedding'].values[news],#embedding
            score = df['probs'].values[news], #prob del tópico
            date = pd.to_datetime(df['start_time_local'].values[news]), #fecha de la noticia
            topico = df['topic'].values[news], #nro tópico al que pertenece
            name_topic = info_por_topico.loc[info_por_topico['Topic'] == df['topic'].values[0],'label'].values[0], #nombre tópico al que pertenece
            sentiment = df['sentiment'].values[news], #nombre sentiment al que pertenece
            score_sentiment =  df['score'].values[news], #prob del tópico
            keywords = [NewsKeyword(name=k) for k in df.iloc[news]['keywords']],#keywords de la noticia
            entities = [NewsEntities(name=k) for k in df.iloc[news]['entities']],#entities de la noticia
            text_doc = df['text'].values[news], #texto del documento 
            index_doc = df['asset_id'].values[news], #index del documento original
            name = df['title'].values[news], #título del documento
            sitio_web = df['Asset Destination'].values[news], #sitio del que se obtuvo el documento
            media = df['media'].values[news], #nombre del medio del que se obtuvo el documento
        )

        print(news_doc.save())
# %%
def get_topic_name(keywords_entidades): #Nombro con los primeros keywords al tópico
    return ', '.join([k for k, s in keywords_entidades[:4]])
# %%
def get_topic_keywords(topic,topic_model, entities_all):
    entities = set(list(zip(*topic_model.topic_representations_[topic]))[0]).intersection(set(entities_all))
    keywords_in_topic = [item for item in list(list(zip(*topic_model.topic_representations_[topic]))[0]) if item not in list(entities)]
    keywords_from_topic = [[item for item in topic_model.topic_representations_[topic] if list(keywords_in_topic)[i] in item] for i in range(len(keywords_in_topic))]
    if(len(keywords_from_topic) > 0):
        topic_keywords = [TopicKeyword(name=k, score=s) for k, s in list(zip(*keywords_from_topic))[0]]
    else: 
        topic_keywords = []
    return topic_keywords
# %%
def get_topic_entities(topic,topic_model, entities_all):
    entities = set(list(zip(*topic_model.topic_representations_[topic]))[0]).intersection(set(entities_all))
    entities_from_topic = [[item for item in topic_model.topic_representations_[topic] if list(entities)[i] in item] for i in range(len(entities))]
    if(len(entities_from_topic) > 0):
        topic_entities = [TopicEntities(name=k, score=s) for k, s in list(zip(*entities_from_topic))[0]]
    else: 
        topic_entities = []
    return topic_entities
# %%
# Guardamos tópicos en la base
def guardar_topicos(topic_model, info_por_topico, sim_matrix, data,title_docs,inicio, date, entities_all):
    for topic in topic_model.get_topics().keys():
        if topic > -1:
            print(topic)
            topic_threshold = info_por_topico[info_por_topico.Topic == topic]['threshold'].values[0]

            best_doc_index = sim_matrix[topic + 1].argmax()
            title_best_doc = title_docs[best_doc_index]
            best_document = data[best_doc_index]        

            topic_doc = Topic(
                vector = list(topic_model.topic_embeddings_[topic + 1]),# vector del tópico, el 0 es -1
                similarity_threshold = topic_threshold, #Sale del análisis previo con histogramas
                created_at = datetime.now(),
                to_date = datetime.strptime(date, '%Y-%m-%d')+timedelta(1),#Agrego un día para que lo testee al siguiente
                from_date = parse(date),
                index = topic+inicio,
                total_docs = info_por_topico[info_por_topico.Topic == topic].docs_sobre_threshold.values[0],
                keywords = get_topic_keywords(topic,topic_model, entities_all),
                entities = get_topic_entities(topic,topic_model, entities_all),
                name = get_topic_name(topic_model.topic_representations_[topic]),
                best_doc = best_document,
                index_best_doc = best_doc_index,
                title_best_doc = title_best_doc,
            )

            print(topic_doc.save())
# %%
def calculo_threshold(info_por_topico, cant_topicos, docs_per_topic_prob):
    for topic in range(cant_topicos):
        if (topic > 0): #No guardo el -1
            topico = topic-1
            mean = np.mean(docs_per_topic_prob[topic]['prob_topic'])
            info_por_topico.loc[info_por_topico.Topic == topico,'threshold'] = mean
            info_por_topico.loc[info_por_topico.Topic == topico,'docs_sobre_threshold'] = docs_per_topic_prob[topic][docs_per_topic_prob[topic]['prob_topic'] >= mean].count().values[0]
# %%
def doc_per_topics(info_por_topico, cant_topicos, df):
    docs_per_topic_prob = {}
    for topic in range(cant_topicos):
        topico = topic-1
        print(f'Tópico :{info_por_topico[info_por_topico.Topic == topico].label.values[0]}')
        docs_per_topic_prob[topic] = df[df['topic'] == topico][['asset_id','title','topic','probs']]
        docs_per_topic_prob[topic].columns = ['id','title_doc','topic','prob_topic']
        docs_per_topic_prob[topic].sort_values(['prob_topic'], inplace = True, ascending = False)
        print(docs_per_topic_prob[topic])
    return docs_per_topic_prob
# %%
def procesamiento(topic_model, data, df, index_docs, title_docs,n_sampleo,inicio_topicos, inicio_noticias, date, entities_all):
    topics, probs = topic_model.fit_transform(data)
    #Embeddings
    embeddings_docs = topic_model.embedding_model.embed(data)
    #Sentiment
    #trabajo con el título de las noticias por el tamaño
    classifier = pipeline('sentiment-analysis',model="nlptown/bert-base-multilingual-uncased-sentiment")
    outputs = classifier(title_docs)
    #Guardo sentiment en el set de datos
    df['sentiment'] = [list(outputs[i].values())[0] for i in range(len(title_docs))]
    df['score'] = [list(outputs[i].values())[1] for i in range(len(title_docs))]
    #Cantidad de tópicos encontrados
    cant_topicos = len(Counter(topics).values())
    #Casos por tópico
    info_por_topico = topic_model.get_topic_freq().sort_values(['Topic'])
    info_por_topico['label'] = topic_model.generate_topic_labels()
    info_por_topico.sort_values(['Count'], inplace = True, ascending = False)
    print(info_por_topico)
    # Guardamos modelo
    topic_model.save(f"modelo_topicos_actualizado")
    #Documentos asociados a tópicos
    df['topic'] = topics
    df['probs'] = probs #Similitud por documento a cada tópico si calculate_probabilities=True, sino una sola prob.
    df['embedding'] = embeddings_docs.tolist()
    guardar_noticias(df,info_por_topico,n_sampleo, inicio_noticias)
    docs_per_topic_prob = doc_per_topics(info_por_topico, cant_topicos, df)
    calculo_threshold(info_por_topico, cant_topicos, docs_per_topic_prob)
    #matriz similitud coseno
    sim_matrix = cosine_similarity(
        topic_model.topic_embeddings_,
        embeddings_docs
    )
    guardar_topicos(topic_model, info_por_topico, sim_matrix, data,title_docs,inicio_topicos, date, entities_all)
# %%
def modelo_topicos(df):
    data = list(df['text']) #analizamos el texto del artículo en vez del título
    index_docs = list(df['asset_id'])
    title_docs = list(df['title'])

    entities_all = set(sum(list([list(e) for e in df['entities'].values]), []))
    keywords_all = set(sum(list([list(e) for e in df['keywords'].values]), []))
    all_tokens = list(entities_all.union(keywords_all))

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
    # Step 1 - Extract embeddings
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    # Step 2 - Reduce dimensionality
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
    # Step 3 - Cluster reduced embeddings
    hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    # Step 4 - Tokenize topics
    vectorizer_model = tf_vectorizer
    # Step 5 - Create topic representation
    ctfidf_model = ClassTfidfTransformer()
    # Step 6 - (Optional) Fine-tune topic representations with 
    # a `bertopic.representation` model
    representation_model = KeyBERTInspired()
    topic_model = BERTopic(
        embedding_model=embedding_model,          # Step 1 - Extract embeddings
        umap_model=umap_model,                    # Step 2 - Reduce dimensionality
        hdbscan_model=hdbscan_model,              # Step 3 - Cluster reduced embeddings
        vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
        ctfidf_model=ctfidf_model,                # Step 5 - Extract topic words
        representation_model=representation_model, # Step 6 - (Optional) Fine-tune topic represenations
        language='spanish',
        #calculate_probabilities=True, #Quiero detectar el principal tópico al que pertenece el documento.
    )
    return topic_model, data, index_docs, title_docs, entities_all
# %%
def inicializar_base():
    Topic.init()
    News.init()
#%%
def borrar_base():
    # #Borramos los tópicos de la base para poder actualizar
    response = os_client.indices.delete(
        index = 'topicos'
    )
    print(response)
