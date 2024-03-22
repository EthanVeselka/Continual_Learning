test.py --tasks 2 --b 500 --ewc --dec      
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
Train: epoch: 1, loss = 0.15551843309774996
Train: epoch: 1, loss = 0.12339548589428886
Train: epoch: 1, loss = 0.10600705337477848
Train: epoch: 1, loss = 0.09892016117111779
Train: epoch: 1, loss = 0.09299541718303225
Train: epoch: 1, loss = 0.09119436572626
Train: epoch: 1, loss = 0.08856811826890668
Train: epoch: 1, loss = 0.08637844067903643
Train: epoch: 1, loss = 0.08699502312455377
Train: epoch: 1, loss = 0.08879407443926902
Train:  Epoch 1, Loss=0.08879407443926902, AUC-ROC=0.8036139376458263, AUC-PR=0.0770099134222998
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.09107843321561813, AUC-ROC=0.8407623011421937, AUC-PR=0.20897152359565263
Eval task: 2
Eval:  Epoch 1, Loss=0.3981247277259827, AUC-ROC=0.6229641443567618, AUC-PR=0.042380491217665045
Experiment dir: ./exp/Test_decomp
------------------------------------
Task: 2, Epoch: 1
Train: epoch: 1, loss = 0.15824695866787805
Train: epoch: 1, loss = 0.14840120436507276
Train: epoch: 1, loss = 0.1445198724853496
Train: epoch: 1, loss = 0.13833140878821723
Train: epoch: 1, loss = 0.13371144799515605
Train: epoch: 1, loss = 0.1335332115398099
Train: epoch: 1, loss = 0.1339263370626473
Train: epoch: 1, loss = 0.13184189385850914
Train: epoch: 1, loss = 0.1315887692790582
Train: epoch: 1, loss = 0.12909846010361797
Train:  Epoch 1, Loss=0.12909846010361797, AUC-ROC=0.6654847790614082, AUC-PR=0.055517324198589305
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.09201130026578903, AUC-ROC=0.8172693896024672, AUC-PR=0.13227727783559823
Eval task: 2
Eval:  Epoch 1, Loss=0.1397777280807495, AUC-ROC=0.7021735255795659, AUC-PR=0.10178600098041633