test.py --tasks 2 --b 1000 --ewc --replay --dec
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
Train: epoch: 1, loss = 0.1374790473189205
Train: epoch: 1, loss = 0.10699362416984513
Train: epoch: 1, loss = 0.0887624704397361
Train: epoch: 1, loss = 0.09041978639565058
Train: epoch: 1, loss = 0.09288480490224901
Train: epoch: 1, loss = 0.0938473573011773
Train: epoch: 1, loss = 0.09186151032427525
Train: epoch: 1, loss = 0.08982827224106586
Train: epoch: 1, loss = 0.08862584271021963
Train: epoch: 1, loss = 0.08559176823386223
Train:  Epoch 1, Loss=0.08559176823386223, AUC-ROC=0.8221789607266817, AUC-PR=0.09777701912073193
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.10258500683307648, AUC-ROC=0.8310337654159603, AUC-PR=0.18443960577200005
Eval task: 2
Eval:  Epoch 1, Loss=0.3788285670280457, AUC-ROC=0.6234302144537044, AUC-PR=0.04163682080319382
Experiment dir: ./exp/Test_decomp
------------------------------------
Task: 2, Epoch: 1
Train: epoch: 1, loss = 0.10842600282281638
Train: epoch: 1, loss = 0.11310060990974308
Train: epoch: 1, loss = 0.10474524732523909
Train: epoch: 1, loss = 0.10158817100571468
Train: epoch: 1, loss = 0.10660734301805497
Train: epoch: 1, loss = 0.10360656394468
Train: epoch: 1, loss = 0.1017620320691328
Train: epoch: 1, loss = 0.09914895710564452
Train: epoch: 1, loss = 0.09932270506112319
Train: epoch: 1, loss = 0.09907081539346836
Train:  Epoch 1, Loss=0.09907081539346836, AUC-ROC=0.6822876691361217, AUC-PR=0.05964067247544001
-------------
Eval task: 1
Eval:  Epoch 1, Loss=0.06163947510719299, AUC-ROC=0.8290089930535058, AUC-PR=0.15669933635720576
Eval task: 2
Eval:  Epoch 1, Loss=0.0918944826722145, AUC-ROC=0.6993837631421524, AUC-PR=0.12424202754567884