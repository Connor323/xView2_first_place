echo "training dpn92 localization model with seeds 0-2"
python train92_loc.py 0 1
python train92_loc.py 1 1
python train92_loc.py 2 1
python tune92_loc.py 0 1
python tune92_loc.py 1 1
python tune92_loc.py 2 1
echo "Model dpn92 localization trained!"