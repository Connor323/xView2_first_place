echo "training resnet34 classification model with seeds 0-2"
python train34_cls.py 0 0
python train34_cls.py 1 0
python train34_cls.py 2 0
python tune34_cls.py 0 0
python tune34_cls.py 1 0
python tune34_cls.py 2 0
echo "Model resnet34 classification trained!"