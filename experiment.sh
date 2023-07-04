for i in {1}
do
    python experiment.py --seed 1 &
    python experiment.py --seed 2 &
    python experiment.py --seed 3 
done