# nanogpt2
GPT2 trained on fineweb edu dataset

## model file 
the model architecture is written in the file `gpt2_fineweb_edu_model.py`
the model can be initialized from this file and weights can be loaded to it

## weights file
the weights file is present inside the `model_weights` directory with name as `gpt2_fineweb_100M_weights_4.pth`

## additional info
- the model is written in pytorch and is trained on 2x Nvidia T4 GPUs from Kaggle.
- it uses tiktoken for tokenization with a vocabulary size of 50304.

## accuracy & prediciton
currently, the model has poor accuracy and prediction with loss of around 5.6 (for decent results it should be around 3.0 if not less). it's blabbering as of now, with some degree of meaningfulness to it.
