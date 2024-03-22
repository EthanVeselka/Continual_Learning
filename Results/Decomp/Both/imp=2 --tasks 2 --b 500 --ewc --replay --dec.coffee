test.py --tasks 2 --b 500 --ewc --replay --dec
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
Train: epoch: 1, loss = 0.15178093994036318
Train: epoch: 1, loss = 0.12380838521989063
Train: epoch: 1, loss = 0.11169153034376601
Train: epoch: 1, loss = 0.10283765765358112
Train: epoch: 1, loss = 0.09895500185957644
Train: epoch: 1, loss = 0.09399391591655634
Train: epoch: 1, loss = 0.0939027451340475
Train: epoch: 1, loss = 0.08961148815091292
Train: epoch: 1, loss = 0.0895167715817095
Train: epoch: 1, loss = 0.08732418557110941
Train:  Epoch 1, Loss=0.08732418557110941, AUC-ROC=0.8087062700093524, AUC-PR=0.0857901697246754
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.08911317205429077, AUC-ROC=0.8238595633069923, AUC-PR=0.20195196071229354
Eval task: 2
Eval:  Epoch 1, Loss=0.3376134400367737, AUC-ROC=0.6575542280240267, AUC-PR=0.07454112736983685
Experiment dir: ./exp/Test_decomp
------------------------------------
Task: 2, Epoch: 1
Train: epoch: 1, loss = 0.10182939028833062
Train: epoch: 1, loss = 0.10632893432863057
Train: epoch: 1, loss = 0.10692078870565941
Train: epoch: 1, loss = 0.10712851191172376
Train: epoch: 1, loss = 0.10425901682581752
Train: epoch: 1, loss = 0.10338997569556038
Train: epoch: 1, loss = 0.10334582616209186
Train: epoch: 1, loss = 0.10244330693007214
Train: epoch: 1, loss = 0.10277554074095355
Train: epoch: 1, loss = 0.1037826211079955
Train:  Epoch 1, Loss=0.1037826211079955, AUC-ROC=0.6814367045345915, AUC-PR=0.061592534197640283
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.07303824666142464, AUC-ROC=0.8150450214916831, AUC-PR=0.17544255635810205
Eval task: 2
Eval:  Epoch 1, Loss=0.09717883104085923, AUC-ROC=0.6981809437765815, AUC-PR=0.10573861646192467