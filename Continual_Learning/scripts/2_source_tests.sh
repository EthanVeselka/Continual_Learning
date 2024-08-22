
# Note: Buffer size is in batches, so num_samples = 8 * buffer_size. Buffer size for all tasks need to be 
# adjusted based on the number of samples per region split. I used a max of 500 for Pheno and IHM for all my 
# results, but since we're excluding so much data that will likely change. 

# Buffer size * 8 <= num samples 
# so you're limited by task 4 which is the smallest (excluding task 5 which we don't sample from cause 
# we don't have a 6th task)

# Once we have the preprocessing completed, my create_splits script in the eICU2MIMIC/ directory will describe the sample 
# count per region when it makes the region splits so we can decide the buffer sizes, and ratios for Decomp and LoS. For 
# any of these, you can add the --pAUC argument to use the pAUC loss instead, results will show up in new_results/.



# These highest priority tests
# All tests are run from the scripts/ dir, comment out what you do not want to run.

# Arguments
# --------------------------------#
# --i is the number of iterations to run, results show the average of all results over --i iterations
# --n limits the number of individual test results shown up to the best <n>, will default to 5 (usual number of iterations)


# IHM: (4 epochs, all samples)   RUN ON GPU 1
# --------------------------------#
python3 ../tests/test.py --region 1 --tasks 2 --ihm --test --rt --i 5                                  # Baseline
python3 ../tests/test.py --region 2 --tasks 2 --ihm --test --rt --i 5                                  # Baseline
python3 ../tests/test.py --region 3 --tasks 2 --ihm --test --rt --i 5                                  # Baseline
python3 ../tests/test.py --region 4 --tasks 2 --ihm --test --rt --i 5                                  # Baseline

python3 ../tests/test.py --region 1 --tasks 2 --ihm --b 500 --replay --ewc --imp 6 --test --rt --i 5   # Combined
python3 ../tests/test.py --region 2 --tasks 2 --ihm --b 500 --replay --ewc --imp 6 --test --rt --i 5   # Combined
python3 ../tests/test.py --region 3 --tasks 2 --ihm --b 500 --replay --ewc --imp 6 --test --rt --i 5   # Combined
python3 ../tests/test.py --region 4 --tasks 2 --ihm --b 500 --replay --ewc --imp 6 --test --rt --i 5   # Combined

python3 ../tests/test.py --region 1 --tasks 2 --ihm --b 500 --replay --test --rt --i 5                 # Adjusted Replay
python3 ../tests/test.py --region 2 --tasks 2 --ihm --b 500 --replay --test --rt --i 5                 # Adjusted Replay
python3 ../tests/test.py --region 3 --tasks 2 --ihm --b 500 --replay --test --rt --i 5                 # Adjusted Replay
python3 ../tests/test.py --region 4 --tasks 2 --ihm --b 500 --replay --test --rt --i 5                 # Adjusted Replay

python3 ../tests/test.py --region 1 --tasks 2 --ihm --b 500 --replay --trrep --test --rt --i 5         # Traditional Replay
python3 ../tests/test.py --region 2 --tasks 2 --ihm --b 500 --replay --trrep --test --rt --i 5         # Traditional Replay
python3 ../tests/test.py --region 3 --tasks 2 --ihm --b 500 --replay --trrep --test --rt --i 5         # Traditional Replay
python3 ../tests/test.py --region 4 --tasks 2 --ihm --b 500 --replay --trrep --test --rt --i 5         # Traditional Replay

python3 ../tests/test.py --region 1 --tasks 2 --ihm --b 500 --ewc --imp 6 --test --rt --i 5            # EWC
python3 ../tests/test.py --region 2 --tasks 2 --ihm --b 500 --ewc --imp 6 --test --rt --i 5            # EWC
python3 ../tests/test.py --region 3 --tasks 2 --ihm --b 500 --ewc --imp 6 --test --rt --i 5            # EWC
python3 ../tests/test.py --region 4 --tasks 2 --ihm --b 500 --ewc --imp 6 --test --rt --i 5            # EWC 


# Phenotyping: (6 epochs, all samples) ALSO RUN ON GPU 1
# -------------------------------- #
python3 ../tests/test.py --region 1 --tasks 2 --phen --test --rt --i 5                                 # Baseline
python3 ../tests/test.py --region 2 --tasks 2 --phen --test --rt --i 5                                 # Baseline
python3 ../tests/test.py --region 3 --tasks 2 --phen --test --rt --i 5                                 # Baseline
python3 ../tests/test.py --region 4 --tasks 2 --phen --test --rt --i 5                                 # Baseline

python3 ../tests/test.py --region 1 --tasks 2 --phen --b 500 --replay --ewc --imp 4 --test --rt --i 5  # Combined (old best)
python3 ../tests/test.py --region 2 --tasks 2 --phen --b 500 --replay --ewc --imp 4 --test --rt --i 5  # Combined (old best)
python3 ../tests/test.py --region 3 --tasks 2 --phen --b 500 --replay --ewc --imp 4 --test --rt --i 5  # Combined (old best)
python3 ../tests/test.py --region 4 --tasks 2 --phen --b 500 --replay --ewc --imp 4 --test --rt --i 5  # Combined (old best)

python3 ../tests/test.py --region 1 --tasks 2 --phen --b 500 --replay --test --rt --i 5                # Adjusted Replay
python3 ../tests/test.py --region 2 --tasks 2 --phen --b 500 --replay --test --rt --i 5                # Adjusted Replay
python3 ../tests/test.py --region 3 --tasks 2 --phen --b 500 --replay --test --rt --i 5                # Adjusted Replay
python3 ../tests/test.py --region 4 --tasks 2 --phen --b 500 --replay --test --rt --i 5                # Adjusted Replay

python3 ../tests/test.py --region 1 --tasks 2 --phen --b 500 --replay --trrep --test --rt --i 5        # Traditional Replay
python3 ../tests/test.py --region 2 --tasks 2 --phen --b 500 --replay --trrep --test --rt --i 5        # Traditional Replay
python3 ../tests/test.py --region 3 --tasks 2 --phen --b 500 --replay --trrep --test --rt --i 5        # Traditional Replay
python3 ../tests/test.py --region 4 --tasks 2 --phen --b 500 --replay --trrep --test --rt --i 5        # Traditional Replay

python3 ../tests/test.py --region 1 --tasks 2 --phen --b 500 --ewc --imp 4 --test --rt --i 5           # EWC
python3 ../tests/test.py --region 2 --tasks 2 --phen --b 500 --ewc --imp 4 --test --rt --i 5           # EWC
python3 ../tests/test.py --region 3 --tasks 2 --phen --b 500 --ewc --imp 4 --test --rt --i 5           # EWC
python3 ../tests/test.py --region 4 --tasks 2 --phen --b 500 --ewc --imp 4 --test --rt --i 5           # EWC 


# Decompensation: (1 epoch, 100k:100k:100k:50k:25k samples)   
# --------------------------------#
python3 ../tests/test.py --region 1 --tasks 2 --dec --test --rt --i 5                                  # Baseline
python3 ../tests/test.py --region 2 --tasks 2 --dec --test --rt --i 5                                  # Baseline
python3 ../tests/test.py --region 3 --tasks 2 --dec --test --rt --i 5                                  # Baseline
python3 ../tests/test.py --region 4 --tasks 2 --dec --test --rt --i 5                                  # Baseline

python3 ../tests/test.py --region 1 --tasks 2 --dec --b 3500 --replay --ewc --imp 6 --test --rt --i 5  # Combined
python3 ../tests/test.py --region 2 --tasks 2 --dec --b 3500 --replay --ewc --imp 6 --test --rt --i 5  # Combined
python3 ../tests/test.py --region 3 --tasks 2 --dec --b 3500 --replay --ewc --imp 6 --test --rt --i 5  # Combined
python3 ../tests/test.py --region 4 --tasks 2 --dec --b 3500 --replay --ewc --imp 6 --test --rt --i 5  # Combined

python3 ../tests/test.py --region 1 --tasks 2 --dec --b 3500 --replay --test --rt --i 5                # Adjusted Replay
python3 ../tests/test.py --region 2 --tasks 2 --dec --b 3500 --replay --test --rt --i 5                # Adjusted Replay
python3 ../tests/test.py --region 3 --tasks 2 --dec --b 3500 --replay --test --rt --i 5                # Adjusted Replay
python3 ../tests/test.py --region 4 --tasks 2 --dec --b 3500 --replay --test --rt --i 5                # Adjusted Replay

python3 ../tests/test.py --region 1 --tasks 2 --dec --b 3500 --replay --trrep --test --rt --i 5        # Traditional Replay
python3 ../tests/test.py --region 2 --tasks 2 --dec --b 3500 --replay --trrep --test --rt --i 5        # Traditional Replay
python3 ../tests/test.py --region 3 --tasks 2 --dec --b 3500 --replay --trrep --test --rt --i 5        # Traditional Replay
python3 ../tests/test.py --region 4 --tasks 2 --dec --b 3500 --replay --trrep --test --rt --i 5        # Traditional Replay

python3 ../tests/test.py --region 1 --tasks 2 --dec --b 3500 --ewc --imp 6 --test --rt --i 5           # EWC
python3 ../tests/test.py --region 2 --tasks 2 --dec --b 3500 --ewc --imp 6 --test --rt --i 5           # EWC
python3 ../tests/test.py --region 3 --tasks 2 --dec --b 3500 --ewc --imp 6 --test --rt --i 5           # EWC
python3 ../tests/test.py --region 4 --tasks 2 --dec --b 3500 --ewc --imp 6 --test --rt --i 5           # EWC


# LoS: (1 epoch, 100k:100k:100k:50k:25k samples)
# --------------------------------#
python3 ../tests/test.py --region 1 --tasks 2 --los --test --rt --i 5                                  # Baseline
python3 ../tests/test.py --region 2 --tasks 2 --los --test --rt --i 5                                  # Baseline
python3 ../tests/test.py --region 3 --tasks 2 --los --test --rt --i 5                                  # Baseline
python3 ../tests/test.py --region 4 --tasks 2 --los --test --rt --i 5                                  # Baseline

python3 ../tests/test.py --region 1 --tasks 2 --los --b 3500 --replay --ewc --imp 6 --test --rt --i 5  # Combined
python3 ../tests/test.py --region 2 --tasks 2 --los --b 3500 --replay --ewc --imp 6 --test --rt --i 5  # Combined
python3 ../tests/test.py --region 3 --tasks 2 --los --b 3500 --replay --ewc --imp 6 --test --rt --i 5  # Combined
python3 ../tests/test.py --region 4 --tasks 2 --los --b 3500 --replay --ewc --imp 6 --test --rt --i 5  # Combined

python3 ../tests/test.py --region 1 --tasks 2 --los --b 3500 --replay --test --rt --i 5                # Adjusted Replay
python3 ../tests/test.py --region 2 --tasks 2 --los --b 3500 --replay --test --rt --i 5                # Adjusted Replay
python3 ../tests/test.py --region 3 --tasks 2 --los --b 3500 --replay --test --rt --i 5                # Adjusted Replay
python3 ../tests/test.py --region 4 --tasks 2 --los --b 3500 --replay --test --rt --i 5                # Adjusted Replay

python3 ../tests/test.py --region 1 --tasks 2 --los --b 3500 --replay --trrep --test --rt --i 5        # Traditional Replay
python3 ../tests/test.py --region 2 --tasks 2 --los --b 3500 --replay --trrep --test --rt --i 5        # Traditional Replay
python3 ../tests/test.py --region 3 --tasks 2 --los --b 3500 --replay --trrep --test --rt --i 5        # Traditional Replay
python3 ../tests/test.py --region 4 --tasks 2 --los --b 3500 --replay --trrep --test --rt --i 5        # Traditional Replay

python3 ../tests/test.py --region 1 --tasks 2 --los --b 3500 --ewc --imp 6 --test --rt --i 5           # EWC
python3 ../tests/test.py --region 2 --tasks 2 --los --b 3500 --ewc --imp 6 --test --rt --i 5           # EWC
python3 ../tests/test.py --region 3 --tasks 2 --los --b 3500 --ewc --imp 6 --test --rt --i 5           # EWC
python3 ../tests/test.py --region 4 --tasks 2 --los --b 3500 --ewc --imp 6 --test --rt --i 5           # EWC



# # LSTM Decompensation: (1 epoch, 100k:100k:100k:50k:25k samples)   
# # --------------------------------#
# python3 ../tests/test.py --region 1 --tasks 2 --dec --test --rt --i 5 --lstm                                  # Baseline
# python3 ../tests/test.py --region 2 --tasks 2 --dec --test --rt --i 5 --lstm                                  # Baseline
# python3 ../tests/test.py --region 3 --tasks 2 --dec --test --rt --i 5 --lstm                                  # Baseline
# python3 ../tests/test.py --region 4 --tasks 2 --dec --test --rt --i 5 --lstm                                  # Baseline

# python3 ../tests/test.py --region 1 --tasks 2 --dec --b 3500 --replay --ewc --imp 6 --test --rt --i 5 --lstm  # Combined
# python3 ../tests/test.py --region 2 --tasks 2 --dec --b 3500 --replay --ewc --imp 6 --test --rt --i 5 --lstm  # Combined
# python3 ../tests/test.py --region 3 --tasks 2 --dec --b 3500 --replay --ewc --imp 6 --test --rt --i 5 --lstm  # Combined
# python3 ../tests/test.py --region 4 --tasks 2 --dec --b 3500 --replay --ewc --imp 6 --test --rt --i 5 --lstm  # Combined

# python3 ../tests/test.py --region 1 --tasks 2 --dec --b 3500 --replay --test --rt --i 5 --lstm                # Adjusted Replay
# python3 ../tests/test.py --region 2 --tasks 2 --dec --b 3500 --replay --test --rt --i 5 --lstm                # Adjusted Replay
# python3 ../tests/test.py --region 3 --tasks 2 --dec --b 3500 --replay --test --rt --i 5 --lstm                # Adjusted Replay
# python3 ../tests/test.py --region 4 --tasks 2 --dec --b 3500 --replay --test --rt --i 5 --lstm                # Adjusted Replay

# python3 ../tests/test.py --region 1 --tasks 2 --dec --b 3500 --replay --trrep --test --rt --i 5 --lstm        # Traditional Replay
# python3 ../tests/test.py --region 2 --tasks 2 --dec --b 3500 --replay --trrep --test --rt --i 5 --lstm        # Traditional Replay
# python3 ../tests/test.py --region 3 --tasks 2 --dec --b 3500 --replay --trrep --test --rt --i 5 --lstm        # Traditional Replay
# python3 ../tests/test.py --region 4 --tasks 2 --dec --b 3500 --replay --trrep --test --rt --i 5 --lstm        # Traditional Replay

# python3 ../tests/test.py --region 1 --tasks 2 --dec --b 3500 --ewc --imp 6 --test --rt --i 5 --lstm           # EWC
# python3 ../tests/test.py --region 2 --tasks 2 --dec --b 3500 --ewc --imp 6 --test --rt --i 5 --lstm           # EWC
# python3 ../tests/test.py --region 3 --tasks 2 --dec --b 3500 --ewc --imp 6 --test --rt --i 5 --lstm           # EWC
# python3 ../tests/test.py --region 4 --tasks 2 --dec --b 3500 --ewc --imp 6 --test --rt --i 5 --lstm           # EWC


# # LSTM LoS: (1 epoch, 100k:100k:100k:50k:25k samples)
# # --------------------------------#
# python3 ../tests/test.py --region 1 --tasks 2 --los --test --rt --i 5 --lstm                                  # Baseline
# python3 ../tests/test.py --region 2 --tasks 2 --los --test --rt --i 5 --lstm                                  # Baseline
# python3 ../tests/test.py --region 3 --tasks 2 --los --test --rt --i 5 --lstm                                  # Baseline
# python3 ../tests/test.py --region 4 --tasks 2 --los --test --rt --i 5 --lstm                                  # Baseline

# python3 ../tests/test.py --region 1 --tasks 2 --los --b 3500 --replay --ewc --imp 6 --test --rt --i 5 --lstm  # Combined
# python3 ../tests/test.py --region 2 --tasks 2 --los --b 3500 --replay --ewc --imp 6 --test --rt --i 5 --lstm  # Combined
# python3 ../tests/test.py --region 3 --tasks 2 --los --b 3500 --replay --ewc --imp 6 --test --rt --i 5 --lstm  # Combined
# python3 ../tests/test.py --region 4 --tasks 2 --los --b 3500 --replay --ewc --imp 6 --test --rt --i 5 --lstm  # Combined

# python3 ../tests/test.py --region 1 --tasks 2 --los --b 3500 --replay --test --rt --i 5 --lstm                # Adjusted Replay
# python3 ../tests/test.py --region 2 --tasks 2 --los --b 3500 --replay --test --rt --i 5 --lstm                # Adjusted Replay
# python3 ../tests/test.py --region 3 --tasks 2 --los --b 3500 --replay --test --rt --i 5 --lstm                # Adjusted Replay
# python3 ../tests/test.py --region 4 --tasks 2 --los --b 3500 --replay --test --rt --i 5 --lstm                # Adjusted Replay

# python3 ../tests/test.py --region 1 --tasks 2 --los --b 3500 --replay --trrep --test --rt --i 5 --lstm        # Traditional Replay
# python3 ../tests/test.py --region 2 --tasks 2 --los --b 3500 --replay --trrep --test --rt --i 5 --lstm        # Traditional Replay
# python3 ../tests/test.py --region 3 --tasks 2 --los --b 3500 --replay --trrep --test --rt --i 5 --lstm        # Traditional Replay
# python3 ../tests/test.py --region 4 --tasks 2 --los --b 3500 --replay --trrep --test --rt --i 5 --lstm        # Traditional Replay

# python3 ../tests/test.py --region 1 --tasks 2 --los --b 3500 --ewc --imp 6 --test --rt --i 5 --lstm           # EWC
# python3 ../tests/test.py --region 2 --tasks 2 --los --b 3500 --ewc --imp 6 --test --rt --i 5 --lstm           # EWC
# python3 ../tests/test.py --region 3 --tasks 2 --los --b 3500 --ewc --imp 6 --test --rt --i 5 --lstm           # EWC
# python3 ../tests/test.py --region 4 --tasks 2 --los --b 3500 --ewc --imp 6 --test --rt --i 5 --lstm           # EWC















