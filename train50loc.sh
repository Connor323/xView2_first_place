echo "training seresnext50 localization model with seeds 0-2"
python train50_loc.py 0 0
python train50_loc.py 1 0
python train50_loc.py 2 0
python tune50_loc.py 0 0
python tune50_loc.py 1 0
python tune50_loc.py 2 0
echo "Model seresnext50 localization trained!"