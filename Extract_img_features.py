# Latent Outlier Exposure for Anomaly Detection with Contaminated Data
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import random 

def initialize_model(model_name, use_pretrained=True):
    model_ft = None
    input_size = 0
    if model_name == "resnet152":
        model_ft = models.resnet152(pretrained=use_pretrained)
        input_size = 224
    return model_ft,input_size

def data_transform(input_size):
    return transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # For FMNIST use this line, Comment out for CIFAR10
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def extract_feature(root):
    model_ft, input_size = initialize_model('resnet152')
    feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1]).to('cuda')
    transform = data_transform(input_size)


    # ## for cifar10
    # trainset = datasets.CIFAR10(root, train=True, transform=transform, download=False)
    # testset = datasets.CIFAR10(root, train=False, transform=transform, download=False)
    ##   for Fmnist
    trainset = datasets.FashionMNIST(root, train=True, transform=transform, download=True)
    testset = datasets.FashionMNIST(root, train=False, transform=transform, download=True)


    train_loader = DataLoader(trainset, batch_size=256,shuffle=False)
    test_loader = DataLoader(testset, batch_size=256,shuffle=False)

    train_features = []
    test_features = []
    train_targets = []
    test_targets = []

    feature_extractor.eval()
    with torch.no_grad():
        for data,target in train_loader:
            data = data.to('cuda')
            feature = feature_extractor(data)
            train_features.append(feature.cpu())
            train_targets.append(target.cpu())
        train_features = torch.cat(train_features,0).squeeze()
        train_targets = torch.cat(train_targets,0)
        for data,target in test_loader:
            data = data.to('cuda')
            feature = feature_extractor(data)
            test_features.append(feature.cpu())
            test_targets.append(target.cpu())

        test_features = torch.cat(test_features,0).squeeze()
        test_targets = torch.cat(test_targets,0)

    return [train_features,train_targets],[test_features,test_targets]


#BULE
def downsample_dataset(root, fraction):
    # Load the original dataset
    trainset = torch.load(root + 'trainset_2048.pt')
    # Unpack the data and labels tensors from the loaded dataset
    data, labels = trainset

    # Calculate the number of samples to keep
    num_samples = int(len(data) * fraction)

    # Randomly select the indices of samples to keep
    random_indices = random.sample(range(len(data)), num_samples)

    # Create a new downsampled dataset
    data_fraction = [data[i] for i in random_indices]
    labels_fraction = [labels[i] for i in random_indices]

    # Combine the data and labels tensors to create the downsampled dataset
    trainset_fraction = [torch.stack(data_fraction), torch.tensor(labels_fraction)]

    # Save the downsampled dataset as a new file
    torch.save(trainset_fraction, root + f'trainset_2048_fraction_{fraction}.pt')



#BULE
def extract_feature_trainfraction(root,fraction):

    """
    get new trainset with fraction of the original trainset but newly extracetd features
    """

    model_ft, input_size = initialize_model('resnet152')
    feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1]).to('cuda')
    transform = data_transform(input_size)


    # ## for cifar10
    # trainset = datasets.CIFAR10(root, train=True, transform=transform, download=False)
    # testset = datasets.CIFAR10(root, train=False, transform=transform, download=False)
    ##   for Fmnist
    trainset = datasets.FashionMNIST(root, train=True, transform=transform, download=True)

    num_samples = int(len(trainset) * fraction)

    sampler = SubsetRandomSampler(torch.randperm(len(trainset))[:num_samples])
    train_loader = DataLoader(trainset, batch_size=256,sampler=sampler)

    train_features = []
    train_targets = []

    feature_extractor.eval()
    with torch.no_grad():
        for data,target in train_loader:
            data = data.to('cuda')
            feature = feature_extractor(data)
            train_features.append(feature.cpu())
            train_targets.append(target.cpu())
        train_features = torch.cat(train_features,0).squeeze()
        train_targets = torch.cat(train_targets,0)


    return [train_features,train_targets]




if __name__=='__main__':

    path = 'DATA'
    trainset, testset = extract_feature(path)
    print("features extracted!")
    # ##CIFAR10 features
    # torch.save(trainset,path+'/cifar10_features/trainset_2048.pt')
    # torch.save(testset, path+'/cifar10_features/testset_2048.pt')

    ##FMNIST features
    torch.save(trainset,path+'/fmnist_features/trainset_2048.pt')
    torch.save(testset, path+'/fmnist_features/testset_2048.pt')

