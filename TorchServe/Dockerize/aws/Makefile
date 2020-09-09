
DOCKER_IP := 34.253.203.131

upmodel:
	aws s3 cp ./../../BERT_deploy/model_store/bert-ner.mar s3://rz-ds-resources/models/bert-ner.mar --profile rz-robot

regmodel:
	curl -X POST "http://$(DOCKER_IP):8081/models?url=https://rz-ds-resources.s3-eu-west-1.amazonaws.com/models/bert-ner.mar&initial_workers=1&synchronous=true&model_name=ner"

models:
	# (Docker from ECS)
	curl http://$(DOCKER_IP):8081/models/

infr:
	curl -X POST http://$(DOCKER_IP):8080/predictions/ner -T ./../../BERT_deploy/SampleText.txt



