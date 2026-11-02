.PHONY: setup train predict evaluate clean

setup:
	pip install --upgrade pip
	pip install -r requirements.txt

train:
	python train.py --epochs 3 --batch-size 16

predict:
	python predict.py

evaluate:
	python -m src.evaluate

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
