#########working code finallllyyyyyyy without early stopping 

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
# import os

# # Check if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Load the cleaned datasets
# train_data_cleaned = pd.read_csv('/home/sai/Downloads/assignment/fully_preprocessed_train.csv')
# test_data_cleaned = pd.read_csv('/home/sai/Downloads/assignment/fully_preprocessed_test.csv')

# # Hyperparameters
# vocab_size = 10000  # Limit vocab_size to top 10,000 most frequent words
# embedding_dim = 50  # Dimension of the embedding vector
# max_length = 136  # Max length of input sequences based on the 95th percentile
# num_classes = len(train_data_cleaned['GENRE'].unique())  # Dynamically get number of classes
# batch_size = 128
# epochs = 500

# # Tokenization
# from tensorflow.keras.preprocessing.text import Tokenizer
# tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
# tokenizer.fit_on_texts(train_data_cleaned['PREPROCESSED_TEXT'])

# # Convert text to sequences
# training_sequences = tokenizer.texts_to_sequences(train_data_cleaned['PREPROCESSED_TEXT'])
# testing_sequences = tokenizer.texts_to_sequences(test_data_cleaned['PREPROCESSED_TEXT'])

# # Padding sequences to ensure uniform length (max_length) and truncate any sequences longer than max_length
# training_padded = torch.tensor(np.array([np.pad(seq, (0, max_length - len(seq))) if len(seq) < max_length else seq[:max_length] for seq in training_sequences]))
# testing_padded = torch.tensor(np.array([np.pad(seq, (0, max_length - len(seq))) if len(seq) < max_length else seq[:max_length] for seq in testing_sequences]))

# # Label Encoding
# label_encoder = LabelEncoder()
# train_labels = torch.tensor(label_encoder.fit_transform(train_data_cleaned['GENRE']))
# test_labels = torch.tensor(label_encoder.transform(test_data_cleaned['GENRE']))

# # Create Dataloaders
# train_data = TensorDataset(training_padded, train_labels)
# test_data = TensorDataset(testing_padded, test_labels)

# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# # Define the model
# class GenreClassifier(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, num_classes):
#         super(GenreClassifier, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm1 = nn.LSTM(embedding_dim, 128, batch_first=True, dropout=0.5)
#         self.lstm2 = nn.LSTM(128, 64, batch_first=True, dropout=0.5)
#         self.fc = nn.Linear(64, num_classes)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.embedding(x)
#         x, (hn, cn) = self.lstm1(x)
#         x, (hn, cn) = self.lstm2(x)
#         x = self.fc(x[:, -1, :])  # Using the output of the last LSTM cell
#         return self.softmax(x)

# # Instantiate the model, loss function, and optimizer
# model = GenreClassifier(vocab_size, embedding_dim, num_classes).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Train the model
# train_accuracy, val_accuracy, train_loss, val_loss = [], [], [], []
# best_val_accuracy = 0

# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
#     correct_preds = 0
#     total_preds = 0
    
#     # Training loop
#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(inputs)
        
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
        
#         # Calculate accuracy
#         _, predicted = torch.max(outputs, 1)
#         correct_preds += (predicted == labels).sum().item()
#         total_preds += labels.size(0)
    
#     train_accuracy.append(correct_preds / total_preds)
#     train_loss.append(running_loss / len(train_loader))
    
#     # Validation loop
#     model.eval()
#     correct_preds = 0
#     total_preds = 0
#     val_running_loss = 0.0
    
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
            
#             loss = criterion(outputs, labels)
#             val_running_loss += loss.item()
            
#             _, predicted = torch.max(outputs, 1)
#             correct_preds += (predicted == labels).sum().item()
#             total_preds += labels.size(0)
    
#     val_accuracy.append(correct_preds / total_preds)
#     val_loss.append(val_running_loss / len(test_loader))
    
#     # Print progress
#     print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy[-1]:.4f}, Val Loss: {val_running_loss/len(test_loader):.4f}, Val Accuracy: {val_accuracy[-1]:.4f}")
    
#     # Save the best model based on validation accuracy
#     if val_accuracy[-1] > best_val_accuracy:
#         best_val_accuracy = val_accuracy[-1]
#         torch.save(model.state_dict(), "best_model.pth")

# # Save the model
# torch.save(model.state_dict(), "genre_classifier.pth")

# # Plot Training Curves (Accuracy & Loss)
# plt.figure(figsize=(12, 5))

# # Accuracy plot
# plt.subplot(1, 2, 1)
# plt.plot(train_accuracy, label='Train Accuracy')
# plt.plot(val_accuracy, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# # Loss plot
# plt.subplot(1, 2, 2)
# plt.plot(train_loss, label='Train Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# # Save the plots
# plt.tight_layout()
# plt.savefig('training_curves_pytorch.png')

# # Show the plots (optional)
# plt.show()

# # Save the training history to CSV
# history_df = pd.DataFrame({
#     'train_accuracy': train_accuracy,
#     'val_accuracy': val_accuracy,
#     'train_loss': train_loss,
#     'val_loss': val_loss
# })
# history_df.to_csv('training_history_pytorch.csv', index=False)


##############new one with early stopiing 



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the cleaned datasets
train_data_cleaned = pd.read_csv('/home/sai/Downloads/assignment/fully_preprocessed_train.csv')
test_data_cleaned = pd.read_csv('/home/sai/Downloads/assignment/fully_preprocessed_test.csv')

# Hyperparameters
vocab_size = 10000  # Limit vocab_size to top 10,000 most frequent words
embedding_dim = 50  # Dimension of the embedding vector
max_length = 136  # Max length of input sequences based on the 95th percentile
num_classes = len(train_data_cleaned['GENRE'].unique())  # Dynamically get number of classes
batch_size = 128
epochs = 500
patience = 5  # Early stopping patience

# Tokenization
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data_cleaned['PREPROCESSED_TEXT'])

# Convert text to sequences
training_sequences = tokenizer.texts_to_sequences(train_data_cleaned['PREPROCESSED_TEXT'])
testing_sequences = tokenizer.texts_to_sequences(test_data_cleaned['PREPROCESSED_TEXT'])

# Padding sequences to ensure uniform length (max_length) and truncate any sequences longer than max_length
training_padded = torch.tensor(np.array([np.pad(seq, (0, max_length - len(seq))) if len(seq) < max_length else seq[:max_length] for seq in training_sequences]))
testing_padded = torch.tensor(np.array([np.pad(seq, (0, max_length - len(seq))) if len(seq) < max_length else seq[:max_length] for seq in testing_sequences]))

# Label Encoding
label_encoder = LabelEncoder()
train_labels = torch.tensor(label_encoder.fit_transform(train_data_cleaned['GENRE']))
test_labels = torch.tensor(label_encoder.transform(test_data_cleaned['GENRE']))

# Create Dataloaders
train_data = TensorDataset(training_padded, train_labels)
test_data = TensorDataset(testing_padded, test_labels)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define the model
class GenreClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(GenreClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, 128, batch_first=True, dropout=0.5)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x, (hn, cn) = self.lstm1(x)
        x, (hn, cn) = self.lstm2(x)
        x = self.fc(x[:, -1, :])  # Using the output of the last LSTM cell
        return self.softmax(x)

# Instantiate the model, loss function, and optimizer
model = GenreClassifier(vocab_size, embedding_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Early stopping setup
best_val_accuracy = 0
best_val_loss = float('inf')
best_f1_score = 0
epochs_without_improvement = 0

# Train the model
train_accuracy, val_accuracy, train_loss, val_loss, f1_scores = [], [], [], [], []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    all_preds = []
    all_labels = []
    
    # Training loop
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    train_accuracy.append(correct_preds / total_preds)
    train_loss.append(running_loss / len(train_loader))
    
    # Calculate F1 score for training data
    train_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Validation loop
    model.eval()
    correct_preds = 0
    total_preds = 0
    val_running_loss = 0.0
    val_all_preds = []
    val_all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
            
            val_all_preds.extend(predicted.cpu().numpy())
            val_all_labels.extend(labels.cpu().numpy())
    
    val_accuracy.append(correct_preds / total_preds)
    val_loss.append(val_running_loss / len(test_loader))
    
    # Calculate F1 score for validation data
    val_f1 = f1_score(val_all_labels, val_all_preds, average='weighted')
    f1_scores.append(val_f1)
    
    # Print progress
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy[-1]:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_running_loss/len(test_loader):.4f}, Val Accuracy: {val_accuracy[-1]:.4f}, Val F1: {val_f1:.4f}")
    
    # Early stopping check (based on F1 score, validation accuracy, and validation loss)
    if val_f1 > best_f1_score or val_accuracy[-1] > best_val_accuracy or val_loss[-1] < best_val_loss:
        best_val_accuracy = val_accuracy[-1]
        best_val_loss = val_loss[-1]
        best_f1_score = val_f1
        torch.save(model.state_dict(), "best_model1.pth")  # Save the best model
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    
    # Stop training if there's no improvement for 'patience' epochs
    if epochs_without_improvement >= patience:
        print("Early stopping triggered")
        break

# Save the model
torch.save(model.state_dict(), "genre_classifier1.pth")

# Plot Training Curves (Accuracy, F1 Score, and Loss)
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# F1 Score plot
plt.subplot(1, 2, 2)
plt.plot(f1_scores, label='Validation F1 Score')
plt.title('Validation F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()

# Save the plots
plt.tight_layout()
plt.savefig('training_curves_pytorch_f1.png')

# Show the plots (optional)
plt.show()

# Save the training history to CSV
history_df = pd.DataFrame({
    'train_accuracy': train_accuracy,
    'val_accuracy': val_accuracy,
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_f1': f1_scores
})
history_df.to_csv('training_history_pytorch_f1.csv', index=False)
