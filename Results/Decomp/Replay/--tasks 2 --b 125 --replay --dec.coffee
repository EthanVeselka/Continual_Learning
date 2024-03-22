test.py --tasks 2 --b 125 --replay --dec
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
Train: epoch: 1, loss = 0.12848037977702917
Train: epoch: 1, loss = 0.11460593718104065
Train: epoch: 1, loss = 0.10223512990322585
Train: epoch: 1, loss = 0.09734297076822258
Train: epoch: 1, loss = 0.09718031418439932
Train: epoch: 1, loss = 0.09205786199386543
Train: epoch: 1, loss = 0.08933108088078111
Train: epoch: 1, loss = 0.08751717665305478
Train: epoch: 1, loss = 0.08824928395760556
Train: epoch: 1, loss = 0.08778832237026654
Train:  Epoch 1, Loss=0.08778832237026654, AUC-ROC=0.7931338204821459, AUC-PR=0.08319009740753933
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.09218529653549194, AUC-ROC=0.8330926236720557, AUC-PR=0.20159286323907277
Eval task: 2
Eval:  Epoch 1, Loss=0.3581293921470642, AUC-ROC=0.6470092640394655, AUC-PR=0.061532311594317934
Experiment dir: ./exp/Test_decomp
------------------------------------
Task: 2, Epoch: 1
Train: epoch: 1, loss = 0.09394663354847581
Train: epoch: 1, loss = 0.09162969795288518
Train: epoch: 1, loss = 0.09508252531755716
Train: epoch: 1, loss = 0.08785653149709105
Train: epoch: 1, loss = 0.08662649782560766
Train: epoch: 1, loss = 0.08678888832529386
Train: epoch: 1, loss = 0.08537044600783182
Train: epoch: 1, loss = 0.08490058762050466
Train: epoch: 1, loss = 0.08342707185294583
Train: epoch: 1, loss = 0.08294199247227516
Train:  Epoch 1, Loss=0.08294199247227516, AUC-ROC=0.6783708835278119, AUC-PR=0.05850871011730565
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.0648218758404255, AUC-ROC=0.78447946272198, AUC-PR=0.11481892339002524
Eval task: 2
Eval:  Epoch 1, Loss=0.07958071249723435, AUC-ROC=0.6928398316821135, AUC-PR=0.08770690894664898
