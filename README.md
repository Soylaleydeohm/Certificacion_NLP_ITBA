# Certificacion_NLP_ITBA

Correr iniciar_modelo.py. Se debe tener descargado modelo_primer_dia_func.py modelos_merged_func.py functions.py opensearch_data_model.py
En las notebooks modelo_primer_dia.py y modelos_merged.py se encuentra la lógica original

Entre las posibles mejoras a futuro, se encuentra lo siguiente:
  
  - Hallar el nuevo mejor documento si es necesario actualizarlo.
  - Considerar una mejor decisión para la actualización de threshold, para evitar que con su cambio documentos que antes pertenecían ahora no deberían pertenecer por el cambio en el umbral. Podría ser usar siempre el mínimo, en vez del promedio.
