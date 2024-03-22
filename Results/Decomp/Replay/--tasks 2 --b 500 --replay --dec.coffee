test.py --tasks 2 --b 500 --replay --dec
--------------------------------------------------
Testing standard LSTM on Decompensation dataset...
--------------------------------------------------


MIMIC training samples: 16000
MIMIC testing samples: 8000
eICU training samples: 16000
eICU testing samples: 8000
Experiment dir: ./exp/Test_decomp
------------------------------------
Task: 1, Epoch: 1
Train: epoch: 1, loss = 0.14570009001530707
Train: epoch: 1, loss = 0.11403360251686535
Train: epoch: 1, loss = 0.09906878189882264
Train: epoch: 1, loss = 0.09931088257988449
Train: epoch: 1, loss = 0.09480201625917106
Train: epoch: 1, loss = 0.09453866750719801
Train: epoch: 1, loss = 0.09287859357893467
Train: epoch: 1, loss = 0.09035578542010626
Train: epoch: 1, loss = 0.0888628940222164
Train: epoch: 1, loss = 0.08690578626049683
Train:  Epoch 1, Loss=0.08690578626049683, AUC-ROC=0.7997913354748172, AUC-PR=0.09769096821760238
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.09307250142097473, AUC-ROC=0.8372444260778874, AUC-PR=0.18242786491795535
Eval task: 2
Eval:  Epoch 1, Loss=0.2018354890346527, AUC-ROC=0.6358589210938205, AUC-PR=0.04820863606753869
Experiment dir: ./exp/Test_decomp
------------------------------------
Task: 2, Epoch: 1
Train: epoch: 1, loss = 0.09630686588818207
Train: epoch: 1, loss = 0.105655658635078
Train: epoch: 1, loss = 0.10013813414688533
Train: epoch: 1, loss = 0.09859825431893114
Train: epoch: 1, loss = 0.09861932781757787
Train: epoch: 1, loss = 0.09944358129170723
Train: epoch: 1, loss = 0.0981937011467692
Train: epoch: 1, loss = 0.09662906661542366
Train: epoch: 1, loss = 0.09703330564493727
Train: epoch: 1, loss = 0.09705381048633717
Train:  Epoch 1, Loss=0.09705381048633717, AUC-ROC=0.6780074277053076, AUC-PR=0.0660099521566718
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.06720937123894691, AUC-ROC=0.8080219217978234, AUC-PR=0.166520983957427
Eval task: 2
Eval:  Epoch 1, Loss=0.11534944832324982, AUC-ROC=0.7051830067769664, AUC-PR=0.11514660942066991
