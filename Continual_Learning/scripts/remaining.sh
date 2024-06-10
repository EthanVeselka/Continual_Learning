# ihm_500_rep_CE.txt
python3 ../tests/test.py --tasks 5 --ihm --b 500 --replay --test --rt --i 5

# phen_<buffsize>_<rep/comb>_CE.txt
python3 ../tests/test.py --tasks 5 --ihm --b 500 --replay --ewc --imp 6 --test --rt --i 5   # Combined (old best)
python3 ../tests/test.py --tasks 5 --ihm --b 500 --replay --test --rt --i 5                 # Replay

sh test_phen_rep.sh  # Replay
sh test_phen_comb.sh # Combined


# Decompensation: (1 epoch, 100k:100k:100k:50k:25k samples)
# --------------------------------#
python3 ../tests/test.py --tasks 5 --dec --test --rt --i 5                                  # Baseline
python3 ../tests/test.py --tasks 5 --dec --b 3500 --replay --ewc --imp 6 --test --rt --i 5  # Combined
python3 ../tests/test.py --tasks 5 --dec --b 3500 --replay --test --rt --i 5                # Replay
python3 ../tests/test.py --tasks 5 --dec --b 3500 --ewc --imp 6 --test --rt --i 5           # EWC (old best)


# LoS: (1 epoch, 100k:100k:100k:50k:25k samples)
# --------------------------------#
python3 ../tests/test.py --tasks 5 --los --test --rt --i 5                                  # Baseline
python3 ../tests/test.py --tasks 5 --los --b 3500 --replay --ewc --imp 6 --test --rt --i 5  # Combined
python3 ../tests/test.py --tasks 5 --los --b 3500 --replay --test --rt --i 5                # Replay
python3 ../tests/test.py --tasks 5 --los --b 3500 --ewc --imp 6 --test --rt --i 5           # EWC


# Decompensation: (Varying Buffer Size Tests)
# These will take multiple days to run
# -------------------------------------#
# sh test_dec_comb.sh # Combined
# sh test_dec_ewc.sh  # EWC
# sh test_dec_rep.sh  # Replay

# LoS: (Varying Buffer Size Tests)
# These will take multiple days to run
# -------------------------------------#
# sh test_los_comb.sh # Combined
# sh test_los_ewc.sh  # EWC
# sh test_los_rep.sh  # Replay