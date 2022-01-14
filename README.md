To run project execute cod from:
https://github.com/pilichm/ZUMNLP_PROJ2/blob/master/colab/Main.ipynb
in google colab.

Used models:
1. GaussainNB,
2. SVC,
3. Logistic Regression,
4. Neural network containing only dense layers:
     embedding (Embedding)       (None, 200, 40)           200000     
     flatten (Flatten)           (None, 8000)              0
     dense (Dense)               (None, 30)                240030     
     dense_1 (Dense)             (None, 20)                620        
     dense_2 (Dense)             (None, 10)                210        
     dense_3 (Dense)             (None, 3)                 33
5. Neural network containing dense and bidirectional LSTM layers:
     embedding_1 (Embedding)     (None, 200, 40)           200000     
     bidirectional (Bidirectiona  (None, 40)               9760
     flatten_1 (Flatten)         (None, 40)                0          
     dense_4 (Dense)             (None, 3)                 123   
6. Neural network containing dense, bidirectional lstm and convolution layers:
      embedding (Embedding)       (None, 200, 40)           200000                                                      
      conv1d (Conv1D)             (None, 195, 20)           4820                                                               
      max_pooling1d (MaxPooling1D  (None, 39, 20)           0                                                              
      bidirectional (Bidirectiona  (None, 40)               6560                                                            
      dense (Dense)               (None, 3)                 123     
7. BERT model.

Trained models are not included, because they exceed guthub file limit.