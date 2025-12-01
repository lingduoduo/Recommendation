[![Testing MLops build](https://github.com/lingduoduo/MLops/actions/workflows/main.yml/badge.svg)](https://github.com/lingduoduo/MLops/actions/workflows/main.yml)

# mlops-sandbox

```
ssh-keygen -t rsa
cat /hone/ec2-user/.ssh/id_rsa.pub
```

```
scann==1.2.6
```

## Setup

```
ssh -i ~/.ssh/ling-ml-dev.pem ec2-user@10.116.110.191@tmg-ml-dev-ssh-gateway-use1@tmg-ml-dev-ssh-gateway-use1.ssh.tmg.luminatesec.com
python3 -m venv ~/.venv 
source ~/.venv/bin/activate
make all
```

## Cli Tools

```
./cli.py --weight 200
./utilscli.py retrain --tsize 0.4
```


## Train a model

* Two Tower Model

```
python src/Recommenders/train_gift_two_tower.py
```

* Two Tower Model uisng SCANN

```
python src/Recommenders/train_gift_scann_retrieval.py
```

* Deep and Cross Network

```
python src/Recommenders/train_gift_dcn_ranking.py
```

* Listwise Ranking

```
python src/Recommenders/train_gift_listwise_ranking.py
```

## Local cli tests

* Two Tower Model

```
./cli.py --jsoninput "kik:user:unknown" "train_gift_two_tower"
```

* Two Tower Model uisng SCANN

```
./cli.py  --jsoninput "tagged:6145378585" "train_gift_scann"
```

* Deep and Cross Network

```
./cli_ranking.py --jsoninput "kik:user:unknown" "kik:user:unknown" "Rose" "10" "train_dcn_ranking"
```

* Listwise Ranking

```
./cli_listwise_ranking.py --jsoninput "kik:user:unknown" "train_listwise_ranking"
```

## Build image
```
docker build -t ling-mlops-sandbox .
```

## List docker images
```
docker image ls
```

## Run flask app
```
docker run -p 127.0.0.1:8080:8080 ling-mlops-sandbox
```

## Prediction
```
./predict.sh
```

## Delete Image if needed
```
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
docker image rm -f ling-mlops-sandbox
```

## Authentication token 
```
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin XXX.dkr.ecr.us-east-1.amazonaws.com
```

## Tag the docker 
```
docker tag ling-mlops-sandbox:latest 882748442234.dkr.ecr.us-east-1.amazonaws.com/ling-mlops-sandbox:latest
```

## Push the images
```
docker push 882748442234.dkr.ecr.us-east-1.amazonaws.com/ling-mlops-sandbox:latest
```

## Pull the images
```
docker pull 882748442234.dkr.ecr.us-east-1.amazonaws.com/ling-mlops-sandbox:latest
```

## Run the predictions
```
docker run -p 127.0.0.1:8080:8080 882748442234.dkr.ecr.us-east-1.amazonaws.com/ling-mlops-sandbox:latest
```
