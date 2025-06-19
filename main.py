import torch

from collections import OrderedDict
from gpt2_fineweb_edu_model import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPT(GPTConfig)
model_state_dict = torch.load("model_weights/gpt2_fineweb_100M_weights_4_corrected.pth", map_location=device)
# model_state_dict = OrderedDict({key.lstrip('_orig_mod.'): value for key, value in model_state_dict.items()})  # fix for original weights (not corrected weights)

model.load_state_dict(model_state_dict)


# generating samples
start_text = "Neural networks today form the base of artificial intelligence"
model.generate_samples(start_text, 5, 50, device)


