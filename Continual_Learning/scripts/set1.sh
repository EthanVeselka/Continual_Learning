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