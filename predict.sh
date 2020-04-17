#!/bin/bash
python predict34_loc.py 0
python predict50_loc.py 0
python predict92_loc.py 0
python predict154_loc.py 0
python predict34cls.py 0 0
python predict50cls.py 0 0
python predict92cls.py 0 0
python predict154cls.py 0 0
python create_submission.py
echo "submission created!"