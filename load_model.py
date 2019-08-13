import pickle
import torch

# with open("model_unsupervised", "rb") as file:
#     if torch.cuda.is_available():
#         model = pickle.load(file=file)
#     else:
#         torch.load(file, map_location={'cuda:0': 'cpu'})

model = torch.load("model_unsupervised.pt", map_location="cpu")

print(model)

for param in model.keys():
    print(param)

# load state_dict or something
