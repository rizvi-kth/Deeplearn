IMAGE_NAME := pytorch/torchserve:rz


upmodel:
	aws s3 cp ./../BERT_deploy/model_store/bert-ner.mar s3://rz-ds-resources/models/bert-ner.mar --profile rz-robot

regmodel:
	curl -X POST "http://localhost:8081/models?url=https://rz-ds-resources.s3-eu-west-1.amazonaws.com/models/bert-ner.mar&initial_workers=1&synchronous=true&model_name=ner"

models:
	# (Docker from ECS)
	curl http://localhost:8081/models/

dock:
	docker run -d --rm -it -p 8080:8080 -p 8081:8081 $(IMAGE_NAME) > /dev/null 2>&1

desc:
	curl http://localhost:8081/models/ner

infr:
	curl -X POST http://127.0.0.1:8080/predictions/ner -T ./../BERT_deploy/SampleText.txt
