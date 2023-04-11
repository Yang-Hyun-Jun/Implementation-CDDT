for i in {1}
do
    python experiment.py --seed 1 &
    python experiment.py --seed 2 &
    python experiment.py --seed 3 
done

for i in {1}
do
    python experiment.py --seed 4 &
    python experiment.py --seed 5 
done

for i in {1}
do
    python experiment.py --seed 1 &
    python experiment.py --seed 2 &
    python experiment.py --seed 3 &
    python experiment.py --seed 4 &
    python experiment.py --seed 5
done