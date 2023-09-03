import torch
import clip
from experiments.utils import *
from models.base_model import ClipDisentangleModel



class CLIPDisentangleExperiment: 
    
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = ClipDisentangleModel(opt)
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        #Setup CLIP model
        self.clip_model, _ = clip.load('ViT-B/32', device='cpu') # load it first to CPU to ensure you're using fp32 precision.
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False #to freeze the clip model

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
        self.clip = opt["weights"][4] 


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
            images = data[0]
            descriptions = data[1]
            images = images.to(self.device)
            descriptions = descriptions.to(self.device)           
            domain_labels = torch.zeros(len(images), dtype=torch.long).to(self.device) 
        else:
            images = data[0]
            images = images.to(self.device)
            domain_labels = torch.ones(len(images), dtype=torch.long).to(self.device) 

        if(len(data) > 2 ): 
            description = data[2]
            tokenized_text = clip.tokenize(description).to(self.device)
            
            text_features = self.clip_model.encode_text(tokenized_text)
            (extractedFeatures, catClass, domClass, advDomClass, advCatClass, reconstructedFeatures, domFeat, clipFeat) = self.model(images, text_features)
        else:
            (extractedFeatures, catClass, domClass, advDomClass, advCatClass, reconstructedFeatures, domFeat, clipFeat) = self.model(images)

        category_loss = 0 if targetDomain == True else self.crossEntropyLoss(catClass, descriptions) 
        
        confuse_domain_loss = -self.entropyLoss(advDomClass)

        domain_loss = self.crossEntropyLoss(domClass, domain_labels)

        confuse_category_loss = -self.entropyLoss(advCatClass)

        reconstructor_loss = self.l2Loss(reconstructedFeatures, extractedFeatures)

        clip_loss = 0 if clipFeat is False else self.l2Loss(domFeat, clipFeat)

        loss = self.weight1*(category_loss + self.alpha*confuse_domain_loss) + self.weight2*(domain_loss + self.alpha*confuse_category_loss) + self.weight3*reconstructor_loss + self.clip*clip_loss
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

                (_, catClass,_, _, _, _, _, _) = self.model(x)
                loss += self.crossEntropyLoss(catClass, y)
                pred = torch.argmax(catClass, dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss

    