import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from experiments.utils import *
from models.base_model import DomainDisentangleModel

class DomainDisentangleExperiment:

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = DomainDisentangleModel(opt)                                      
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Optimization
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.crossEntropyLoss = torch.nn.CrossEntropyLoss()    
        self.entropyLoss = EntropyLoss()
        self.l2Loss = L2Loss() 

        # Weights
        self.weight1 = opt["weights"][0]
        self.weight2 = opt["weights"][1]
        self.weight3 = opt["weights"][2]
        self.alpha = opt["weights"][3]
        print("Domain Disentangle parameters: \n","weight1: ", self.weight1, "weight2: ", self.weight2, "weight3: ", self.weight3, "alpha: ", self.alpha)

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, best_accuracy, total_train_loss

    def train_iteration(self, data, targetDomain = False):

        self.optimizer.zero_grad()  
        
        if(targetDomain == False):
            images, y = data 
            images = images.to(self.device)
            labels = labels.to(self.device)           
            domain_labels = torch.zeros(len(images), dtype=torch.long).to(self.device) 
        else:
            images, _ = data 
            images = images.to(self.device)
            domain_labels = torch.ones(len(images), dtype=torch.long).to(self.device) 

        (extractedFeatures, catClass, domClass, advDomClass, advCatClass, reconstructedFeatures, _) = self.model(images) 

        category_loss = 0 if targetDomain == True else self.crossEntropyLoss(catClass, labels)
        
        confuse_domain_loss = -self.entropyLoss(advDomClass)           

        domain_loss = self.crossEntropyLoss(domClass, domain_labels)    

        confuse_category_loss = -self.entropyLoss(advCatClass)

        reconstructor_loss = self.l2Loss(reconstructedFeatures, extractedFeatures)

        loss = self.weight1*(category_loss + self.alpha*confuse_domain_loss) + self.weight2*(domain_loss + self.alpha*confuse_category_loss) + self.weight3*reconstructor_loss
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, loader): 
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
            for data in loader:
                x = data[0]
                y = data[1]
                x = x.to(self.device)
                y = y.to(self.device)

                (_, catClass, _, _, _, _, _) = self.model(x)
                loss += self.crossEntropyLoss(catClass, y)
                pred = torch.argmax(catClass, dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss
