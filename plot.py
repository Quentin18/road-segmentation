import matplotlib.pyplot as plt
import pickle


with open('losses.pickle', 'rb') as f:
    data = pickle.load(f)
train_loss, test_loss = data['train_loss'], data['test_loss']
plt.style.use("ggplot")
plt.figure()
plt.plot(train_loss, label="train_loss")
plt.plot(test_loss, label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.show()
# plt.savefig(config.PLOT_PATH)
