test.py --tasks 2 --b 1000 --ewc --dec
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
Train: epoch: 1, loss = 0.1196822607005015
Train: epoch: 1, loss = 0.10832656474551186
Train: epoch: 1, loss = 0.09396290447213687
Train: epoch: 1, loss = 0.0915873293209006
Train: epoch: 1, loss = 0.09169286848255433
Train: epoch: 1, loss = 0.09135761395620648
Train: epoch: 1, loss = 0.09120239432518637
Train: epoch: 1, loss = 0.0885680917405989
Train: epoch: 1, loss = 0.08896957531338558
Train: epoch: 1, loss = 0.0880031695375219
Train:  Epoch 1, Loss=0.0880031695375219, AUC-ROC=0.7990444587368981, AUC-PR=0.08064755237043
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.09268778604269028, AUC-ROC=0.8389571543830946, AUC-PR=0.19681593711269524
Eval task: 2
Eval:  Epoch 1, Loss=0.2430795738697052, AUC-ROC=0.6184317407052307, AUC-PR=0.04032768912399377
Experiment dir: ./exp/Test_decomp
------------------------------------
Task: 2, Epoch: 1
Train: epoch: 1, loss = 0.14575052785221487
Train: epoch: 1, loss = 0.13767267416929826
Train: epoch: 1, loss = 0.13511085365200415
Train: epoch: 1, loss = 0.1283169154456118
Train: epoch: 1, loss = 0.12815271431813016
Train: epoch: 1, loss = 0.126977204985839
Train: epoch: 1, loss = 0.12596728700512488
Train: epoch: 1, loss = 0.12721323978999863
Train: epoch: 1, loss = 0.12796302585008865
Train: epoch: 1, loss = 0.12711241989652627
Train:  Epoch 1, Loss=0.12711241989652627, AUC-ROC=0.6725469563924847, AUC-PR=0.061953796171472314
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.09376768285036087, AUC-ROC=0.8117003871033035, AUC-PR=0.1342712951992752
Eval task: 2
Eval:  Epoch 1, Loss=0.13184524738788606, AUC-ROC=0.7077556112791012, AUC-PR=0.11521987907738254