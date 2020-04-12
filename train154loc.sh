echo "training senet154 localization model with seeds 0-2"
python train154_loc.py 0 1
python train154_loc.py 1 1
python train154_loc.py 2 1
echo "Model senet154 localization trained!"