
env:
	pipenv shell

mar:
	torch-model-archiver  --model-name bert-ner  \
	                        --version 1.0  \
	                        --serialized-file ./../../NLP/BERT/test1/models/bert_CoNLL_model_colab_2.pth  \
	                        --handler ./NerClassifierHandler.py

	mv bert-ner.mar ./model_store/

# --ts-config to set a configuration file to load from
# --ncs(--no-config-snapshots) stop generating new snapshot configurations (don't use first time starting a service)
# check https://github.com/pytorch/serve/blob/master/docs/snapshot.md
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
