# Docker containerization of a model.mar

### Create a custom docker image

You need to create a custom docker image if you need some package i.e.`transformers` for your model to work.
  
    1. Edit the `Dockerfile` to include the package.

You need to have the whole repository downloaded on the directory where the `Dockerfile` is. Either `git clone https://github.com/pytorch/serve.git` or download it.   
    
    2. Download the `pytorch/serve` repository in the `serve` folder.

Run `build_image.sh` and check the image has successfully created by `docker images`. Use `DOCKER_BUILDKIT` to build the image; Use custom tag i.e. `pytorch/torchserve:rz`. 

    3. Run `build_image.sh` to prepare the new image. 

Before starting the container and register the model - make sure that the model.mar file is in place. 

#### For local model registration

  -  Make sure the model `model.mar` is in location `./model_store/`. 
  -  You need to start the container with volume-mount with the location. 
  -  Finally in the CURL command specify _only the file name_ `model.mar` in the `url` parameter.

#### For model registration from S3 

  -  Upload the `model.mar` in the S3 bucket and give (public) access. 
  -  Start the container without volume mount.  
  -  Finally in the CURL command specify _S3 url to the file_ `model.mar` in the `url` parameter.


You can modify the `start.sh` file for _local_ or _S3_ models accordingly.

Now run the `start.sh` to run the container and **register** the `model.mar` file and set _config parameters_. 
   
    4. Run `start.sh` to run the container and register model with config.

