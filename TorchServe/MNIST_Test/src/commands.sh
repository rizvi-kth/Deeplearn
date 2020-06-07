#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Provide the step number you want to execute as cli parameter."
fi

if [ "$1" == 1 ]; then
  torch-model-archiver  --model-name mnist  \
                        --version 1.0  \
                        --model-file ./mnist.py  \
                        --serialized-file ./mnist_cnn.pt  \
                        --handler ./mnist_handler.py
fi

if [ "$1" == 2 ]; then
  torchserve --start --model-store model_store --models mnist=mnist.mar

  # [Docker] Use this command to start TorchServe from a Docker container.
  # Start TorchServe with snapshot feature disabled. You can use --ncs flag for this.
  # torchserve --ncs --start --model-store model_store --models mnist=mnist.mar

fi

if [ "$1" == 3 ]; then
  curl -X POST http://127.0.0.1:8080/predictions/mnist -T test_data/0.png
fi


if [ "$1" == 99 ]; then
  docker run --rm -it -p 8080:8080 -p 8081:8081 pytorch/torchserve:latest bash

  # Running Model Archiver Option 1
  docker run --rm -p 8080:8080 -p 8081:8081 -v $(pwd)/model-store:/model-store  -v $(pwd)/examples:/examples torchserve:latest bash
  model-server@98b1c682955c:~$ torch-model-archiver ...

  # Running Model Archiver Option 2
  docker run --rm -p 8080:8080 -p 8081:8081 -v $(pwd)/model-store:/model-store  -v $(pwd)/examples:/examples torchserve:latest torch-model-archiver --model-name densenet161 --version 1.0 --model-file /examples/image_classifier/densenet_161/model.py --serialized-file /examples/image_classifier/densenet161-8d451a50.pth --export-path /model-store --extra-files /examples/image_classifier/index_to_name.json --handler image_classifier

  docker run --rm -it -p 8080:8080 -p 8081:8081 -v $(pwd)/model_store:/model-store pytorch/torchserve:latest

  docker cp ./mnist.mar bbb3ca6da87f:/home/model-server/model-store

fi

# torchserve --stop

