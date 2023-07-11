import pandas as pd
import matplotlib.pyplot as plt
from util import *
from model import *
from train import train_epoch

df = pd.read_csv("archive/questions.csv")
# print(df.shape)

max_length = 50
q_pair, labels = transform_data(df[:100000], max_length)
# print(q_pair[0].shape)


batch_size = 128
train_loader, test_loader = dataset_generator(q_pair, labels, batch_size, 0.2, max_length)

# dataset_iter = iter(train_loader)
# temp = next(dataset_iter)
# feature1, feature2, labels = temp
# print(feature1.shape, feature2.shape, labels.shape)


threshold = torch.Tensor([0.5]).cuda()  # threshold for determining similiarity
learning_rate = 0.0001
epochs = 50
hidden_dim = 100
embedding_dim = max_length
num_layers = 3

model = LSTM(embedding_dim, hidden_dim, num_layers, dropout=0).cuda()
loss_fn = nn.MSELoss()
siamese = SiameseNet(batch_size, model).cuda()
optimizer = torch.optim.Adam(siamese.parameters(), lr=learning_rate)

loss = train_epoch(epochs, train_loader,
                siamese, optimizer, loss_fn, threshold)

# print(loss)
plt.plot(loss)
plt.show()






