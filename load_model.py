import pickle

with open("model_unsupervised", "rb") as file:
    model = pickle.load(file=file)

for param in model.embedding_net.convnet.parameters():
    print(param)

# load state_dict or something
