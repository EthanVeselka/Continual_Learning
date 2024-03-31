Notes:

IHM: 28 epochs (best basis at 4) (bad overfitting past this)
- 14,681 mimic training samples
- 3,222 mimic val samples
- 50,758 eICU training samples
- 10,877 eICU val samples

Phen: 20 epochs (best basis at 6) (haven't tried replay/ewc)
- 29,250 mimic training samples
- 6,371 mimic val samples
- 33,684 eICU training samples
- 7,218 eICU val samples

Decomp: 36 chunks (2:1 tr/val ratio) ~300k samples per task
- 2377768 mimic training samples
- 530646 mimic val samples
- 1,275,068 eICU training samples
- 274,197 eICU val samples

LoS: 19 chunks (2:1 tr/val ratio) ~150000 samples per task
- 2391740 mimic training samples
- 533694 mimic val samples
- 1,274,068 eICU training samples
- 274,197 eICU val samples

Buffer: 
- How do we want to weight data in buffer (percentage composition by each task)?
+ equal distr, not random (label each sample in some way)
- What buffer sizes do we want to test, as a proportion of total samples being trained on?
+ I need to figure it out (something like 0 - 50 % of training sample size)

script model testing (after all training, report test results for each task)