test.py --tasks 2 --b 1000 --ewc --replay --dec
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
Train: epoch: 1, loss = 0.12772454779362305
Train: epoch: 1, loss = 0.11511694223503582
Train: epoch: 1, loss = 0.10070024342470181
Train: epoch: 1, loss = 0.09383352480290341
Train: epoch: 1, loss = 0.09334577246627304
Train: epoch: 1, loss = 0.091208429740412
Train: epoch: 1, loss = 0.08909116823201267
Train: epoch: 1, loss = 0.08849734119379718
Train: epoch: 1, loss = 0.08705987961012095
Train: epoch: 1, loss = 0.08669493668567156
Train:  Epoch 1, Loss=0.08669493668567156, AUC-ROC=0.815149497542102, AUC-PR=0.0901140563739201
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.08835067361593246, AUC-ROC=0.8458558581352135, AUC-PR=0.2017650724022629
Eval task: 2
Eval:  Epoch 1, Loss=0.36553997564315793, AUC-ROC=0.6450717440650326, AUC-PR=0.056694038820901224
Experiment dir: ./exp/Test_decomp
------------------------------------
Task: 2, Epoch: 1
Train: epoch: 1, loss = 0.09196716025238856
Train: epoch: 1, loss = 0.10160131787532009
Train: epoch: 1, loss = 0.09868556149148693
Train: epoch: 1, loss = 0.09335580526269041
Train: epoch: 1, loss = 0.09491038713697345
Train: epoch: 1, loss = 0.09575155105597029
Train: epoch: 1, loss = 0.09591765407206757
Train: epoch: 1, loss = 0.09697517389198765
Train: epoch: 1, loss = 0.09681594772160881
Train: epoch: 1, loss = 0.09688732408639043
Train:  Epoch 1, Loss=0.09688732408639043, AUC-ROC=0.6727051308192302, AUC-PR=0.055035196305173964
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.07343205952644348, AUC-ROC=0.8247573927735095, AUC-PR=0.18391391160841766
Eval task: 2
Eval:  Epoch 1, Loss=0.08165845900774002, AUC-ROC=0.6926034675615211, AUC-PR=0.08234380260998973