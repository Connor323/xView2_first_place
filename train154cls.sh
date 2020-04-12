echo "training senet154 classification model with seeds 0-2"
python train154_cls_cce.py 0 1
python train154_cls_cce.py 1 1
python train154_cls_cce.py 2 1
python tune154_cls_cce.py 0 1
python tune154_cls_cce.py 1 1
python tune154_cls_cce.py 2 1
echo "Model senet154 classification trained!"