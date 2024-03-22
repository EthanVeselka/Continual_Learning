test.py --tasks 2 --b 500 --ewc --dec      
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
Train: epoch: 1, loss = 0.13173292676918208
Train: epoch: 1, loss = 0.09826418758020736
Train: epoch: 1, loss = 0.09032653603237123
Train: epoch: 1, loss = 0.09147721095127054
Train: epoch: 1, loss = 0.08997855580155738
Train: epoch: 1, loss = 0.08994810158134593
Train: epoch: 1, loss = 0.08791979801009542
Train: epoch: 1, loss = 0.0871908787076245
Train: epoch: 1, loss = 0.08656820267619979
Train: epoch: 1, loss = 0.08643433422758244
Train:  Epoch 1, Loss=0.08643433422758244, AUC-ROC=0.8065179946826508, AUC-PR=0.08931038391643271
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.0879704082608223, AUC-ROC=0.8445823810367171, AUC-PR=0.1928628885212501
Eval task: 2
Eval:  Epoch 1, Loss=0.46193847370147706, AUC-ROC=0.6363618670665651, AUC-PR=0.05653195455639382
Experiment dir: ./exp/Test_decomp
------------------------------------
Task: 2, Epoch: 1
Train: epoch: 1, loss = 0.17388309424743056
Train: epoch: 1, loss = 0.15251450040144846
Train: epoch: 1, loss = 0.1413140807603486
Train: epoch: 1, loss = 0.1340317873726599
Train: epoch: 1, loss = 0.13512241303641348
Train: epoch: 1, loss = 0.13367811147123576
Train: epoch: 1, loss = 0.13431029459262
Train: epoch: 1, loss = 0.13130704192386475
Train: epoch: 1, loss = 0.13169136872204643
Train: epoch: 1, loss = 0.12936991673661397
Train:  Epoch 1, Loss=0.12936991673661397, AUC-ROC=0.6655285015121198, AUC-PR=0.05767395370147101
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.0925500255227089, AUC-ROC=0.8155517416837448, AUC-PR=0.1264874011527875
Eval task: 2
Eval:  Epoch 1, Loss=0.13647921133041382, AUC-ROC=0.7084058046725832, AUC-PR=0.09272680799283908
