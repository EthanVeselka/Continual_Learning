# Decompensation: (1 epoch, 100k:100k:100k:50k:25k samples)   
# --------------------------------#
python3 ../tests/test.py --region 1 --tasks 2 --dec --test --rt --i 5 --lstm                                  # Baseline
python3 ../tests/test.py --region 2 --tasks 2 --dec --test --rt --i 5 --lstm                                  # Baseline
python3 ../tests/test.py --region 3 --tasks 2 --dec --test --rt --i 5 --lstm                                  # Baseline
python3 ../tests/test.py --region 4 --tasks 2 --dec --test --rt --i 5 --lstm                                  # Baseline

python3 ../tests/test.py --region 1 --tasks 2 --dec --b 3500 --replay --ewc --imp 6 --test --rt --i 5 --lstm  # Combined
python3 ../tests/test.py --region 2 --tasks 2 --dec --b 3500 --replay --ewc --imp 6 --test --rt --i 5 --lstm  # Combined
python3 ../tests/test.py --region 3 --tasks 2 --dec --b 3500 --replay --ewc --imp 6 --test --rt --i 5 --lstm  # Combined
python3 ../tests/test.py --region 4 --tasks 2 --dec --b 3500 --replay --ewc --imp 6 --test --rt --i 5 --lstm  # Combined

python3 ../tests/test.py --region 1 --tasks 2 --dec --b 3500 --replay --test --rt --i 5 --lstm                # Adjusted Replay
python3 ../tests/test.py --region 2 --tasks 2 --dec --b 3500 --replay --test --rt --i 5 --lstm                # Adjusted Replay
python3 ../tests/test.py --region 3 --tasks 2 --dec --b 3500 --replay --test --rt --i 5 --lstm                # Adjusted Replay
python3 ../tests/test.py --region 4 --tasks 2 --dec --b 3500 --replay --test --rt --i 5 --lstm                # Adjusted Replay

python3 ../tests/test.py --region 1 --tasks 2 --dec --b 3500 --replay --trrep --test --rt --i 5 --lstm        # Traditional Replay
python3 ../tests/test.py --region 2 --tasks 2 --dec --b 3500 --replay --trrep --test --rt --i 5 --lstm        # Traditional Replay
python3 ../tests/test.py --region 3 --tasks 2 --dec --b 3500 --replay --trrep --test --rt --i 5 --lstm        # Traditional Replay
python3 ../tests/test.py --region 4 --tasks 2 --dec --b 3500 --replay --trrep --test --rt --i 5 --lstm        # Traditional Replay

python3 ../tests/test.py --region 1 --tasks 2 --dec --b 3500 --ewc --imp 6 --test --rt --i 5 --lstm           # EWC
python3 ../tests/test.py --region 2 --tasks 2 --dec --b 3500 --ewc --imp 6 --test --rt --i 5 --lstm           # EWC
python3 ../tests/test.py --region 3 --tasks 2 --dec --b 3500 --ewc --imp 6 --test --rt --i 5 --lstm           # EWC
python3 ../tests/test.py --region 4 --tasks 2 --dec --b 3500 --ewc --imp 6 --test --rt --i 5 --lstm           # EWC
