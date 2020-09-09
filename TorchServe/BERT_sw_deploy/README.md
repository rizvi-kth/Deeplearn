## Steps to get go

Run the commands from the `Makefile`

  -  `pipenv shell` 
  
  Make sure that TorchServe is already installed in the Environment.
  
  -  Run `torch-model-archiver` to generate .mar file.
  
  Make sure you have the proper handler-class writen for the model.
  
  -  Start the Torch Server with `torchserve`
  
  - Check the model with `curl` url test.
   