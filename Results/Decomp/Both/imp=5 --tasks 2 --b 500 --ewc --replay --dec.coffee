test.py --tasks 2 --b 500 --ewc --replay --dec
--------------------------------------------------
Testing standard LSTM on Decompensation dataset...
--------------------------------------------------

importance = 5

MIMIC training samples: 16000
MIMIC testing samples: 8000
eICU training samples: 16000
eICU testing samples: 8000
Experiment dir: ./exp/Test_decomp
------------------------------------
Task: 1, Epoch: 1
Train: epoch: 1, loss = 0.13567653943784536
Train: epoch: 1, loss = 0.11317097802646459
Train: epoch: 1, loss = 0.10498265268436323
Train: epoch: 1, loss = 0.10162911162129604
Train: epoch: 1, loss = 0.09691934313368984
Train: epoch: 1, loss = 0.09356136661042304
Train: epoch: 1, loss = 0.09098756062282648
Train: epoch: 1, loss = 0.09004537813088972
Train: epoch: 1, loss = 0.08889945209285037
Train: epoch: 1, loss = 0.08812452365993523
Train:  Epoch 1, Loss=0.08812452365993523, AUC-ROC=0.7998978334613891, AUC-PR=0.08155821298082977
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.09084681171178818, AUC-ROC=0.8277467607859013, AUC-PR=0.18762319988893048
Eval task: 2
Eval:  Epoch 1, Loss=0.2679593150615692, AUC-ROC=0.6284970622218944, AUC-PR=0.05074044217427012
Experiment dir: ./exp/Test_decomp
------------------------------------
Task: 2, Epoch: 1
Train: epoch: 1, loss = 0.12136528557632119
Train: epoch: 1, loss = 0.10872805594466627
Train: epoch: 1, loss = 0.10798148627858609
Train: epoch: 1, loss = 0.10530846096517052
Train: epoch: 1, loss = 0.10302004996920004
Train: epoch: 1, loss = 0.10203690525493585
Train: epoch: 1, loss = 0.09934142622098859
Train: epoch: 1, loss = 0.09989450263587059
Train: epoch: 1, loss = 0.09888208026506214
Train: epoch: 1, loss = 0.09950257329270244
Train:  Epoch 1, Loss=0.09950257329270244, AUC-ROC=0.682900612307711, AUC-PR=0.0682092328263397
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.08729421031475067, AUC-ROC=0.819708112302126, AUC-PR=0.18715008496739968
Eval task: 2
Eval:  Epoch 1, Loss=0.11683076298236847, AUC-ROC=0.6905627463513371, AUC-PR=0.11496225054350366