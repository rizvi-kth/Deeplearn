#!/bin/bash
IMAGE_NAME="pytorch/torchserve:rz"


echo "Starting docker image : $IMAGE_NAME"
echo "======================="

# For S3 Model
# docker run -d --rm -it -p 8080:8080 -p 8081:8081 $IMAGE_NAME > /dev/null 2>&1

# For local Model
docker run -d --rm -it -p 8080:8080 -p 8081:8081 -v $(pwd)/../BERT_deploy/model_store:/home/model-server/model-store $IMAGE_NAME > /dev/null 2>&1

container_id=$(docker ps --filter="ancestor=$IMAGE_NAME" -q | xargs)

sleep 30

echo "Successfully started torchserve in docker."
echo "Container ID : $container_id"




echo ""
echo ""
echo "Registering bert-ner.mar model"
echo "=============================="
# S3 Models
# curl -X POST "http://localhost:8081/models?url=https://rz-ds-resources.s3-eu-west-1.amazonaws.com/bert-ner.mar&initial_workers=1&synchronous=true&model_name=ner"
# response=$(curl --write-out %{http_code} --silent --output /dev/null --retry 5 -X POST "http://localhost:8081/models?url=https://rz-ds-resources.s3-eu-west-1.amazonaws.com/bert-ner.mar&initial_workers=1&synchronous=true&model_name=ner")

# Local model
# curl -X POST "http://localhost:8081/models?url=bert-ner.mar&initial_workers=1&synchronous=true&model_name=ner"
# response=$(curl --write-out %{http_code} --silent --output /dev/null --retry 5 -X POST "http://localhost:8081/models?url=bert-ner.mar&initial_workers=1&synchronous=true&model_name=ner")

if [ ! "$response" == 200 ]
then
    echo "failed to register model with torchserve"
else
    echo "successfully registered ner model with torchserve"
fi




echo ""
echo ""
echo "TorchServe is up and running with ner model"
echo "==========================================="
echo "Management APIs are accessible on http://127.0.0.1:8081"
echo "Inference APIs are accessible on http://127.0.0.1:8080"
echo "To stop docker container for TorchServe use command : docker container stop $container_id"
