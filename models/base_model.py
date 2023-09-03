import torch
import torch.nn as nn
from torch import cat
from torchvision.models import resnet18

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet18 = resnet18(pretrained=True)
    
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        x.squeeze()
        if len(x.size())<2:
            x.unsqueeze(0)
        return x

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.classifier = nn.Linear(512, 7)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.category_encoder(x)
        x = self.classifier(x)
        return x

class DomainDisentangleModel(nn.Module):
    def __init__(self, opt):
        super(DomainDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor() 
        self.opt = opt
        if opt['encoder_type'] == 'conv':
        
            self.domain_encoder = nn.Sequential(

                nn.Conv1d(1,256,kernel_size=3,padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),

                nn.Conv1d(256,256,kernel_size=3,padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),

                nn.Conv1d(256,256,kernel_size=3,padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),

                nn.Conv1d(256,1,kernel_size=3,padding=1),
                nn.BatchNorm1d(1),
                nn.ReLU()

            )

            self.category_encoder = nn.Sequential(

                nn.Conv1d(1,256,kernel_size=3,padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),

                nn.Conv1d(256,256,kernel_size=3,padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),

                nn.Conv1d(256,256,kernel_size=3,padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),

                nn.Conv1d(256,1,kernel_size=3,padding=1),
                nn.BatchNorm1d(1),
                nn.ReLU()

            )

        elif opt['encoder_type'] == 'linear':

            self.domain_encoder = nn.Sequential(      

                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU()

            )

        
            self.category_encoder = nn.Sequential(
                
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),

                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU()

            )

        
        #Domain
        self.domain_classifier = nn.Linear(512, 2)      

        #Category
        self.category_classifier = nn.Linear(512, 7)    


        if opt['reconstructor_type'] == 'conv':
        
            self.reconstructor = nn.Sequential(

                nn.Conv1d(1,256,kernel_size=3,stride=2,padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),

                nn.Conv1d(256,256,kernel_size=3,padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),

                nn.Conv1d(256,256,kernel_size=3,padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),

                nn.Conv1d(256,1,kernel_size=3,padding=1),
                nn.BatchNorm1d(1),
                nn.ReLU()

            )
            

        elif opt['reconstructor_type'] == 'linear':

            self.reconstructor = nn.Sequential(             
                                                            
            nn.Linear(1024, 512),                          
            nn.BatchNorm1d(512),                            
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()

        )


    def forward(self, x):

        #Feature Extractor
        extractedFeatures = self.feature_extractor(x)     

        if self.opt['reconstructor_type'] == 'linear':      
          extractedFeatures = extractedFeatures.squeeze(2)
        extractedFeatures = extractedFeatures.squeeze(2)
        if self.opt['reconstructor_type'] == 'conv':
          extractedFeatures = torch.transpose(extractedFeatures,1,2)
        
        #Features
        domFeat = self.domain_encoder(extractedFeatures)     
        catFeat = self.category_encoder(extractedFeatures)   

        # N x F x H
        if self.opt['reconstructor_type'] == 'conv':
          domFeat = torch.transpose(domFeat,1,2).squeeze(2)     
          catFeat = torch.transpose(catFeat,1,2).squeeze(2)
        
        #Train
        domClass = self.domain_classifier(domFeat)                                                
        catClass = self.category_classifier(catFeat) 
        
        #Adversarial
        advDomClass = self.domain_classifier(catFeat)  
        advCatClass = self.category_classifier(domFeat)  


        concatFeatures = cat((catFeat, domFeat), 1)
        if self.opt['reconstructor_type'] == 'conv':
          concatFeatures = torch.transpose(concatFeatures.unsqueeze(2),1,2)

        #Reconstructor  
        reconstructedFeatures = self.reconstructor(concatFeatures) 
        if self.opt['reconstructor_type'] == 'conv':
          reconstructedFeatures = torch.transpose(reconstructedFeatures,1,2).squeeze(2)
        
        return (extractedFeatures, catClass, domClass, advDomClass, advCatClass, reconstructedFeatures, domFeat)

class ClipDisentangleModel(nn.Module):
    def __init__(self, opt):
        super(ClipDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()

        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        #Domain
        self.domain_classifier = nn.Linear(512, 2) 
        #Category
        self.category_classifier = nn.Linear(512, 7) 

        self.CLIP_fullyconnected = nn.Linear(512, 512)

        self.reconstructor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )


    def forward(self, x, clip_features = False):
        
        #Feature Extractor
        extractedFeatures = self.feature_extractor(x) 

        extractedFeatures = extractedFeatures.squeeze(2)
        extractedFeatures = extractedFeatures.squeeze(2)

        #Features
        catFeat = self.category_encoder(extractedFeatures)    
        domFeat = self.domain_encoder(extractedFeatures)      

        #Train
        catClass = self.category_classifier(catFeat)  
        domClass = self.domain_classifier(domFeat) 

        #Adversarial
        advDomClass = self.domain_classifier(catFeat)  
        advCatClass = self.category_classifier(domFeat) 

        #Reconstructor
        reconstructedFeatures = self.reconstructor(cat((catFeat, domFeat), 1)) 
        
        if clip_features is not False: 
            clipFeat = self.CLIP_fullyconnected(clip_features)
        else:
            clipFeat = False

        return (extractedFeatures, catClass, domClass, advDomClass, advCatClass, reconstructedFeatures, domFeat, clipFeat)