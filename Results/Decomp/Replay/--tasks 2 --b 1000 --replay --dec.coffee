test.py --tasks 2 --b 1000 --replay --dec
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
Train: epoch: 1, loss = 0.1507191641256213
Train: epoch: 1, loss = 0.1268605002516415
Train: epoch: 1, loss = 0.10896145592598866
Train: epoch: 1, loss = 0.10357827067084145
Train: epoch: 1, loss = 0.10016814113990404
Train: epoch: 1, loss = 0.09601126014313195
Train: epoch: 1, loss = 0.09401223988970742
Train: epoch: 1, loss = 0.0894410290975793
Train: epoch: 1, loss = 0.08936664216343261
Train: epoch: 1, loss = 0.08731882275774842
Train:  Epoch 1, Loss=0.08731882275774842, AUC-ROC=0.7962590197933264, AUC-PR=0.09930774004237007
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.08870301073789597, AUC-ROC=0.8371678206674369, AUC-PR=0.20057988947203212
Eval task: 2
Eval:  Epoch 1, Loss=0.28913265037536623, AUC-ROC=0.6394133457891846, AUC-PR=0.054579807291415317
Experiment dir: ./exp/Test_decomp
------------------------------------
Task: 2, Epoch: 1
Train: epoch: 1, loss = 0.10166010198649018
Train: epoch: 1, loss = 0.09974528087419458
Train: epoch: 1, loss = 0.1001937987228545
Train: epoch: 1, loss = 0.10127511859929655
Train: epoch: 1, loss = 0.1007415126026608
Train: epoch: 1, loss = 0.10184401370934211
Train: epoch: 1, loss = 0.10058668478624895
Train: epoch: 1, loss = 0.10084423527790932
Train: epoch: 1, loss = 0.10121061155313833
Train: epoch: 1, loss = 0.10135906869266183
Train:  Epoch 1, Loss=0.10135906869266183, AUC-ROC=0.6814031356387843, AUC-PR=0.0610627376412211
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.09463981479406357, AUC-ROC=0.8257746985682518, AUC-PR=0.15581725550653952
Eval task: 2
Eval:  Epoch 1, Loss=0.10841000032424927, AUC-ROC=0.7092373044554252, AUC-PR=0.13175242287682723