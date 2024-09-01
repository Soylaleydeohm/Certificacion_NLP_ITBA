from opensearchpy import Float, OpenSearch, Field, Integer, Document, Keyword, Text, DenseVector, Nested, Date, Object, connections, InnerDoc
import os

# local test
# docker pull opensearchproject/opensearch:latest
# docker run -it -p 9200:9200 -p 9600:9600 -e OPENSEARCH_INITIAL_ADMIN_PASSWORD=PassWord#1234! -e "discovery.type=single-node"  --name opensearch-node opensearchproject/opensearch:latest
# docker stop opensearch-node
# docker start opensearch-node



OPENSEARCH_HOST = os.getenv('OPENSEARCH_HOST', "localhost")
auth = ('admin', 'PassWord#1234!')
port = 9200
os_client = connections.create_connection(
    hosts = [{'host': OPENSEARCH_HOST, 'port': port}],
    http_auth = auth,
    http_compress = True, # enables gzip compression for request bodies
    use_ssl = True,
    verify_certs = False,
    alias='default'
    # ssl_assert_hostname = False,
    # ssl_show_warn = False
)

TOPIC_DIMENSIONS = 384
TOPIC_INDEX_NAME = 'topicos'
NEWS_INDEX_NAME = 'noticias'
TOPIC_INDEX_PARAMS = {
    'number_of_shards': 1,
    'knn': True
}

knn_params = {
    "name": "hnsw",
    "space_type": "cosinesimil",
    "engine": "nmslib"
}

class TopicKeyword(InnerDoc):
    name = Keyword()
    score = Float()

class NewsKeyword(InnerDoc):
    name = Keyword()

class TopicEntities(InnerDoc):
    name = Keyword()
    score = Float()

class NewsEntities(InnerDoc):
    name = Keyword()

class SimilarTopics(Document):
    topic_id = Keyword()
    similar_to = Keyword()
    similarity = Float()
    common_keywwords = Keyword()
    keywords_not_in_similar = Keyword()
    keywords_not_in_topic = Keyword()

class KNNVector(Field):
    name = "knn_vector"
    def __init__(self, dimension, method, **kwargs):
        super(KNNVector, self).__init__(dimension=dimension, method=method, **kwargs)

class Topic(Document):
    vector = KNNVector(TOPIC_DIMENSIONS, knn_params) #vector
    similarity_threshold = Float() #umbral del tópico
    created_at = Date() #fecha en la que apareció el tópico por primera vez
    to_date = Date()
    from_date = Date()
    index = Integer() #nro tópico
    name = Text() #nombre tópico
    total_docs = Integer() #cantidad de documentos con el tópico
    entities = Object(TopicEntities) #keywords del tópico  
    keywords = Object(TopicKeyword) #entities del tópico    
    best_doc = Text() #texto del documento más representativo del tópico
    index_best_doc = Text() #index del documento más representativo del tópico
    title_best_doc = Text() #título del documento más representativo del tópico
    
    class Index:
        name = TOPIC_INDEX_NAME
        if not os_client.indices.exists(index=TOPIC_INDEX_NAME):
            settings = {
                'index': TOPIC_INDEX_PARAMS
            }

    def save(self, ** kwargs):
        self.meta.id = f'{self.index}' + self.name.replace(', ', '-').replace(' ', '_')
        return super(Topic, self).save(** kwargs)


class News(Document):
    index = Integer() #nro noticia
    vector = KNNVector(TOPIC_DIMENSIONS, knn_params) #embedding
    score = Float() #prob del tópico
    date = Date() #fecha de la noticia
    topico = Integer() #nro tópico al que pertenece
    name_topic = Text() #nombre tópico al que pertenece
    sentiment = Text() #nombre sentiment al que pertenece
    score_sentiment =  Float() #prob del tópico
    keywords = Object(NewsKeyword) #keywords de la noticia
    entities = Object(NewsEntities) #entities de la noticia
    text_doc = Text() #texto del documento 
    index_doc = Text() #index del documento 
    name = Text() #título del documento
    sitio_web = Text() #sitio del que se obtuvo el documento
    media = Text() #nombre del medio del que se obtuvo el documento
    
    class Index:
        name = NEWS_INDEX_NAME
        if not os_client.indices.exists(index=NEWS_INDEX_NAME):
            settings = {
                'index': TOPIC_INDEX_PARAMS
            }

    def save(self, ** kwargs):
        self.meta.id = f'{self.index}' + self.name.replace(', ', '-').replace(' ', '_')
        return super(News, self).save(** kwargs)