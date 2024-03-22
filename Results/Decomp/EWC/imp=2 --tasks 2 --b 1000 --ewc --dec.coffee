test.py --tasks 2 --b 1000 --ewc --dec      
--------------------------------------------------
Testing standard LSTM on Decompensation dataset...
--------------------------------------------------

importance = 2

MIMIC training samples: 16000
MIMIC testing samples: 8000
eICU training samples: 16000
eICU testing samples: 8000
Experiment dir: ./exp/Test_decomp
------------------------------------
Task: 1, Epoch: 1
Train: epoch: 1, loss = 0.11827353907283396
Train: epoch: 1, loss = 0.11377495135180653
Train: epoch: 1, loss = 0.10491720048167433
Train: epoch: 1, loss = 0.09845578085369197
Train: epoch: 1, loss = 0.09625232252315619
Train: epoch: 1, loss = 0.0909057376467778
Train: epoch: 1, loss = 0.0908295574251263
Train: epoch: 1, loss = 0.08974877119595476
Train: epoch: 1, loss = 0.0889180980561327
Train: epoch: 1, loss = 0.08787656346400036
Train:  Epoch 1, Loss=0.08787656346400036, AUC-ROC=0.8019798094917919, AUC-PR=0.08318542247997411
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.08962735509872437, AUC-ROC=0.841694567903317, AUC-PR=0.18643269663374687
Eval task: 2
Eval:  Epoch 1, Loss=0.22187846398353578, AUC-ROC=0.6325649322713081, AUC-PR=0.04388288524936758
Experiment dir: ./exp/Test_decomp
------------------------------------
Task: 2, Epoch: 1
Train: epoch: 1, loss = 0.112435041324934
Train: epoch: 1, loss = 0.11486809885886032
Train: epoch: 1, loss = 0.11783440197080684
Train: epoch: 1, loss = 0.12705690322007285
Train: epoch: 1, loss = 0.12858410403621384
Train: epoch: 1, loss = 0.12773640459345187
Train: epoch: 1, loss = 0.13034242019539566
Train: epoch: 1, loss = 0.12886821091189632
Train: epoch: 1, loss = 0.12576305943088503
Train: epoch: 1, loss = 0.12638167834177147
Train:  Epoch 1, Loss=0.12638167834177147, AUC-ROC=0.6808896558621791, AUC-PR=0.06394097128378814
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.09435321980714798, AUC-ROC=0.789319659618969, AUC-PR=0.12844394327298997
Eval task: 2
Eval:  Epoch 1, Loss=0.13063934648036957, AUC-ROC=0.6981115454269817, AUC-PR=0.12435230058667139