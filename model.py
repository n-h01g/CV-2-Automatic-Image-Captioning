import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers = 1):
        
        super(DecoderRNN, self).__init__()
        
        # embedding layer to turn words into a vector of a specified size
        self.embeddings = nn.Embedding(num_embeddings = vocab_size, 
                                       embedding_dim = embed_size)

        # LSTM to take embedded word vectors (of a specified size) as inputs and outputs hidden states of size
        self.lstm = nn.LSTM(input_size = embed_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers,
                            bias = True,
                            batch_first = True,
                            dropout = 0,
                            bidirectional = False)

        # linear layer to map the hidden state output dimension 
        self.linear = nn.Linear(in_features = hidden_size, 
                                out_features = vocab_size)        
  
    def forward(self, features, captions):
        
        # create embedded word vectors for each word in a sentence
        captions = captions[:, :-1]
        captions = self.embeddings(captions)
        
        # concatenate the given sequence
        features = features.unsqueeze(1)
        inputs = torch.cat((features, captions), dim = 1)
        
        # get the output by passing the lstm over word embeddings
        lstm_out, _ = self.lstm(inputs)
        
        # convert LSTM outputs to sentences
        outputs = self.linear(lstm_out)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        result = []
        
        # for each tensor
        for i in range(max_len):
            
            # get the output and hidden state by passing the lstm over word embeddings
            lstm_out, states = self.lstm(inputs, states)
            
            # convert LSTM outputs to sentences
            outputs = self.linear(lstm_out)
            
            # reduce dimensionality
            outputs = outputs.squeeze(1)
            
            # get word with max probability value
            word = outputs.max(1)[1]
            
            # append word to list to create sentence
            result.append(word.item())
            
            # embed last predicted word for next iteration into lstm
            inputs = self.embeddings(word).unsqueeze(1)
            
        return result