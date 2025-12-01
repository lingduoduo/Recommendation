install:
	pip3 install --upgrade pip && pip3 install -r requirements.txt

test:
	python -m pytest -vv --cov=main test_*.py

format:
	black $$(git ls-files '*.py')

#src/train_gift_two_tower.py
#src/train_gift_dcn_ranking.py
lint:
#	pylint --disable=R,C $$(git ls-files '*.py')
	pylint --disable=R,C main.py

deploy:
	echo "deploy command goes here"

all: install lint test format deploy
