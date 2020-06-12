
env:
	pipenv shell

mar:
	torch-model-archiver  --model-name bert-ner  \
	                        --version 1.0  \
	                        --serialized-file ./../../NLP/BERT/test1/models/bert_CoNLL_model_colab_2.pth  \
	                        --handler ./NerClassifierHandler.py

	mv bert-ner.mar ./model_store/

# --ts-config to set a configuration file to load from
# --ncs(--no-config-snapshots) stop generating new snapshot configurations
# https://github.com/pytorch/serve/blob/master/docs/snapshot.md
# https://pytorch.org/serve/configuration.html#config-properties-file
# First time run :  You can create a snapshot configuration by not using --ncs argument;
#                   this create a snapshot configuration in ./logs/config/
# Later time run :  Prepare config.properties from the snapshot configuration in ./logs/config/ then start TS server
start:
	torchserve --start --model-store model_store --models ner=bert-ner.mar --ncs --ts-config ./logs/config/20200606162654498-shutdown.cfg

stop:
	torchserve --stop

clean:
	rm ./model_store/*

srv: mar start



models:
	curl http://localhost:8081/models/


desc:
	curl http://localhost:8081/models/ner

infr:
	curl -X POST http://127.0.0.1:8080/predictions/ner -T SampleText.txt

# (Docker steps)
# docker run --rm -it -p 8080:8080 -p 8081:8081 --name ner -v $(pwd)/model_store:/home/model-server/model-store pytorch/torchserve:latest
# docker exec -it <cont-id> bash
# torchserve stop
# pip install transformers
# torchserve --start --model-store model_store --models ner=bert-ner.mar --ncs
# docker cp ./config.properties <cont-id>:/home/model-server/       (keep NUM_WORKERS=1 for local docker image)


