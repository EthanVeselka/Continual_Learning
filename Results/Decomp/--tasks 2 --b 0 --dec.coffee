test.py --tasks 2 --b 0 --dec
--------------------------------------------------
Testing standard LSTM on Decompensation dataset...
--------------------------------------------------


NOTE: Buffer size is 0, EWC and Replay will not be used
MIMIC training samples: 16000
MIMIC testing samples: 8000
eICU training samples: 16000
eICU testing samples: 8000
Experiment dir: ./exp/Test_decomp
------------------------------------
Task: 1, Epoch: 1
Train: epoch: 1, loss = 0.1402748685516417
Train: epoch: 1, loss = 0.11083217814622913
Train: epoch: 1, loss = 0.10084999003098347
Train: epoch: 1, loss = 0.09576739201293094
Train: epoch: 1, loss = 0.09529132914007642
Train: epoch: 1, loss = 0.09299863230495248
Train: epoch: 1, loss = 0.09236568201516222
Train: epoch: 1, loss = 0.09036976120609325
Train: epoch: 1, loss = 0.08932623293069708
Train: epoch: 1, loss = 0.08747841919446364
Train:  Epoch 1, Loss=0.08747841919446364, AUC-ROC=0.7991975495925956, AUC-PR=0.09207757590056853
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.09276086375117303, AUC-ROC=0.8298653572061094, AUC-PR=0.1926683311647637
Eval task: 2
Eval:  Epoch 1, Loss=0.3427015900611877, AUC-ROC=0.6562891806180396, AUC-PR=0.07010538773423923
Experiment dir: ./exp/Test_decomp
------------------------------------
Task: 2, Epoch: 1
Train: epoch: 1, loss = 0.136189896191936
Train: epoch: 1, loss = 0.13062213959288782
Train: epoch: 1, loss = 0.13364405444279934
Train: epoch: 1, loss = 0.13165141440462322
Train: epoch: 1, loss = 0.13050078321620823
Train: epoch: 1, loss = 0.12992239073462164
Train: epoch: 1, loss = 0.12846593879362833
Train: epoch: 1, loss = 0.12807888180192095
Train: epoch: 1, loss = 0.1283783779717568
Train: epoch: 1, loss = 0.12710013668285683
Train:  Epoch 1, Loss=0.12710013668285683, AUC-ROC=0.679585718382659, AUC-PR=0.06474749892152615
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.09316373455524445, AUC-ROC=0.7890807069625182, AUC-PR=0.15001629197317748
Eval task: 2
Eval:  Epoch 1, Loss=0.1339803307056427, AUC-ROC=0.7040255140087356, AUC-PR=0.09321080665583008