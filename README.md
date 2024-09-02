# Certificacion_NLP_ITBA

Para el primer día emplear modelo_primer_dia.py

Luego, para agregar más noticias, emplear modelos_merged.py

Entre las posibles mejoras a futuro, se encuentra lo siguiente:

  Opté en no utilizar el método merge_models de BERTopic, y en cambio calcular sólo los casos en que no podía hallar un tópico válido en los días previos. Esto genera que luego sea necesario cargar cada uno de los modelos previos si se corre varias veces. 
  
  No se recalculó el threshold de los días anteriores, aunque dejo guardado en la base la cantidad de documentos asociados a un tópico para que sea más sencillo este cálculo en otra etapa.
