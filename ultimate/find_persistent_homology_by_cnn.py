#!/usr/bin/env python
# coding: utf-8

# # Preservation of Topology with Deep Image Segementation
# 
# ## Debayan Datta

# In[123]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

from sklearn.metrics import adjusted_rand_score
from scipy.stats import entropy

import pandas as pd
import matplotlib.pyplot as plt

import itertools
from math import comb

from PIL import Image
import h5py

import torchvision.models as models


# ## Original Neural Network

# In[82]:


# Define the neural network
class TopologyPreservingCNN(nn.Module):
    def __init__(self):
        super(TopologyPreservingCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)  # Assuming input patches of size 65x65
        self.fc2 = nn.Linear(1024, 2)  # Assuming binary classification for segmentation

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool4(F.relu(self.conv4(x)))
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# In[83]:


# Instantiate the model
model = TopologyPreservingCNN()
print(model)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# ## Random sampling of patches from an image

# In[84]:


# Function to sample patches
def random_patch_sampling(image, patch_size=65, num_patches=100):
    patches = []
    img_height, img_width = image.shape
    for _ in range(num_patches):
        top_left_y = np.random.randint(0, img_height - patch_size)
        top_left_x = np.random.randint(0, img_width - patch_size)
        patch = image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size]
        patches.append(patch)
    return patches


# ## Persistence Homology

# In[85]:


def compute_persistence_2DImg_1DHom(f):
    """
    compute persistence diagram in a 2D function (can be N-dim) and critical pts
    only generate 1D homology dots and critical points
    """
    assert len(f.shape) == 2  # f has to be 2D function
    dim = 2

    # pad the function with a few pixels of minimum values
    # this way one can compute the 1D topology as loops
    # remember to transform back to the original coordinates when finished
    padwidth = 2
    padvalue = min(f.min(), 0.0)
    f_padded = np.pad(f, padwidth, 'constant', constant_values=padvalue)
    
    #print("about to call cubePers")
    
    # call persistence code to compute diagrams
    # loads PersistencePython.so (compiled from C++); should be in current dir
    from PersistencePython import cubePers
    #print("cubePers imported success")
    persistence_result = cubePers(np.reshape(
        f_padded, f_padded.size).tolist(), list(f_padded.shape), 0.001)
    #print("Persistence result calculated")

    persistence_result_filtered = [k for k in persistence_result if k[0]==1]
    persistence_result_filtered = np.array(persistence_result_filtered)
    
    # persistence diagram (second and third columns are coordinates)
    dgm = persistence_result_filtered[:, 1:3]
    
    # critical points
    birth_cp_list = persistence_result_filtered[:, 4:4 + dim]
    death_cp_list = persistence_result_filtered[:, 4 + dim:]

    # when mapping back, shift critical points back to the original coordinates
    birth_cp_list = birth_cp_list - padwidth
    death_cp_list = death_cp_list - padwidth
    
    return dgm, birth_cp_list, death_cp_list


# In[86]:


def compute_dgm_force(lh_dgm, gt_dgm):
    # get persistence list from both diagrams
    lh_pers = lh_dgm[:, 1] - lh_dgm[:, 0]
    gt_pers = gt_dgm[:, 1] - gt_dgm[:, 0]

    # more lh dots than gt dots
    assert lh_pers.size > gt_pers.size

    # check to ensure that all gt dots have persistence 1
    tmp = gt_pers > 0.999
    assert tmp.sum() == gt_pers.size
    
    gt_n_holes = gt_pers.size  # number of holes in gt

    # get "perfect holes" - holes which do not need to be fixed, i.e., find top
    # lh_n_holes_perfect indices
    # check to ensure that at least one dot has persistence 1; it is the hole
    # formed by the padded boundary
    # if no hole is ~1 (ie >.999) then just take all holes with max values
    tmp = lh_pers > 0.999  # old: assert tmp.sum() >= 1
    if sum(tmp) >= 1:
        # n_holes_to_fix = gt_n_holes - lh_n_holes_perfect
        lh_n_holes_perfect = tmp.sum()
        idx_holes_perfect = np.argpartition(lh_pers, -lh_n_holes_perfect)[
                            -lh_n_holes_perfect:]
    else:
        idx_holes_perfect = np.where(lh_pers == lh_pers.max())[0]

    # find top gt_n_holes indices
    idx_holes_to_fix_or_perfect = np.argpartition(lh_pers, -gt_n_holes)[
                                  -gt_n_holes:]

    # the difference is holes to be fixed to perfect
    idx_holes_to_fix = list(
        set(idx_holes_to_fix_or_perfect) - set(idx_holes_perfect))

    # remaining holes are all to be removed
    idx_holes_to_remove = list(
        set(range(lh_pers.size)) - set(idx_holes_to_fix_or_perfect))

    # only select the ones whose persistence is large enough
    # set a threshold to remove meaningless persistence dots
    # TODO values below this are small dents so dont fix them; tune this value?
    pers_thd = 0.03
    idx_valid = np.where(lh_pers > pers_thd)[0]
    idx_holes_to_remove = list(
        set(idx_holes_to_remove).intersection(set(idx_valid)))

    force_list = np.zeros(lh_dgm.shape)
    # push each hole-to-fix to (0,1)
    force_list[idx_holes_to_fix, 0] = 0 - lh_dgm[idx_holes_to_fix, 0]
    force_list[idx_holes_to_fix, 1] = 1 - lh_dgm[idx_holes_to_fix, 1]

    # push each hole-to-remove to (0,1)
    force_list[idx_holes_to_remove, 0] = lh_pers[idx_holes_to_remove] / \
                                         math.sqrt(2.0)
    force_list[idx_holes_to_remove, 1] = -lh_pers[idx_holes_to_remove] / \
                                         math.sqrt(2.0)

    return force_list, idx_holes_to_fix, idx_holes_to_remove


# In[87]:


def compute_topological_loss(lh_dgm, gt_dgm):
    
    force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(lh_dgm, gt_dgm)
    loss = 0.0
    for idx in idx_holes_to_fix:
        loss += force_list[idx, 0] ** 2 + force_list[idx, 1] ** 2
    for idx in idx_holes_to_remove:
        loss += force_list[idx, 0] ** 2 + force_list[idx, 1] ** 2
    
    return loss


# In[88]:


def compute_topological_grad(lh_dgm, lh_bcp, lh_dcp, gt_dgm):
    
    force_list, idx_holes_to_fix, idx_holes_to_remove = compute_dgm_force(lh_dgm, gt_dgm)
    topo_grad = np.zeros([2 * (len(idx_holes_to_fix) + len(idx_holes_to_remove)), 3])
    counter = 0
    for idx in idx_holes_to_fix:
        topo_grad[counter] = [lh_bcp[idx, 1], lh_bcp[idx, 0], force_list[idx, 0]]
        counter += 1
        topo_grad[counter] = [lh_dcp[idx, 1], lh_dcp[idx, 0], force_list[idx, 1]]
        counter += 1
    for idx in idx_holes_to_remove:
        topo_grad[counter] = [lh_bcp[idx, 1], lh_bcp[idx, 0], force_list[idx, 0]]
        counter += 1
        topo_grad[counter] = [lh_dcp[idx, 1], lh_dcp[idx, 0], force_list[idx, 1]]
        counter += 1
    topo_grad[:, 2] = topo_grad[:, 2] * -2
    
    return topo_grad


# ### some more functions

# In[89]:


def compute_probabilities_labels(tensor):
    probabilities = []
    
    for i in range(tensor.size(0)):  # Iterate over the batch size dimension
        array = tensor[i]  # Get the i-th array of shape (65, 65)
        flat_array = array.view(-1) # Flatten the array to make counting easier
        
        # Calculate the no. and probabilities of True and False 
        num_true = torch.sum(flat_array)
        num_false = flat_array.numel() - num_true
        prob_true = num_true.float() / flat_array.numel()
        prob_false = num_false.float() / flat_array.numel()
        
        probabilities.append([prob_true.item(), prob_false.item()])
    
    probabilities_tensor = torch.tensor(probabilities)
    
    return probabilities_tensor


# In[150]:


def rgb_to_grayscale(image_path):
    # Open the image file
    img = Image.open(image_path).convert('RGB')  # Ensure image is in RGB mode
    # Convert the image to grayscale
    grayscale_img = img.convert('L')
    # Convert the grayscale image to a numpy array
    grayscale_array = np.array(grayscale_img)
    # Normalize the array to have values between 0 and 1
    normalized_array = grayscale_array / 255.0
    return normalized_array


# ## Metrics

# In[90]:


# PER PIXEL ACCURACY

def calculate_pixel_accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)  
    
    if labels.dim() > 1:
        labels = labels.argmax(dim=1)  
    correct = preds.eq(labels).sum().item()  
    total = labels.numel()  
    return correct / total


# In[91]:


# BETTI NUMBER ERROR 

def calculate_betti_error(lh, gt):
    betti_error = np.abs(len(lh) - len(gt))
    return betti_error 


# In[92]:


# VARIATION OF INFORMATION

def calculate_voi(x, y): 
    x1 = torch.argmax(x, dim=1).cpu().numpy() # Extract the predicted classes from x
    y1 = torch.argmax(y, dim=1).cpu().numpy() # Extract the actual classes from y (one-hot encoding)
    contingency_table = np.histogram2d(x1, y1, bins=(2, 2))[0] 
    joint_prob = contingency_table / contingency_table.sum() 

    marginal_prob_preds = joint_prob.sum(axis=1)
    marginal_prob_labels = joint_prob.sum(axis=0)

    mutual_info = np.sum(joint_prob * np.log(joint_prob / (marginal_prob_preds[:, None] * marginal_prob_labels[None, :] + 1e-10) + 1e-10))
    voi = entropy(marginal_prob_preds) + entropy(marginal_prob_labels) - 2 * mutual_info 
    return voi


# In[93]:


# ADJUSTED RAND INDEX

def calculate_rand_index_manual(outputs, labels):
    predicted_classes = torch.argmax(outputs, dim=1)
    actual_classes = torch.argmax(labels, dim=1)
    N = outputs.size(0)
    a = 0; b = 0
    for i, j in itertools.combinations(range(N), 2):
        same_cluster_pred = predicted_classes[i] == predicted_classes[j]
        same_cluster_actual = actual_classes[i] == actual_classes[j]
        if same_cluster_pred and same_cluster_actual:
            a += 1
        elif not same_cluster_pred and not same_cluster_actual:
            b += 1
    total_pairs = comb(N, 2)
    RI = (a + b) / total_pairs   
    return RI, predicted_classes, actual_classes

def calculate_ari_manual(outputs, labels):
    RI, predicted_classes, actual_classes = calculate_rand_index_manual(outputs, labels)
    N = outputs.size(0)
    contingency_matrix = np.zeros((2, 2))
    for i in range(N):
        contingency_matrix[actual_classes[i], predicted_classes[i]] += 1
    sum_comb_c1 = sum(comb(int(n), 2) for n in contingency_matrix.sum(axis=1))
    sum_comb_c2 = sum(comb(int(n), 2) for n in contingency_matrix.sum(axis=0))
    total_pairs = comb(N, 2)
    E = (sum_comb_c1 * sum_comb_c2) / total_pairs
    max_RI = 1.0
    ARI = (RI - E) / (max_RI - E)    
    return ARI


# # Input

# In[151]:


def list_hdf5_contents(file_path):
    with h5py.File(file_path, 'r') as file:
        def print_attrs(name, obj):
            print(f"{name}: {type(obj)}")
            if isinstance(obj, h5py.Dataset):
                print(f"  - Shape: {obj.shape}")
                print(f"  - Data type: {obj.dtype}")        
        file.visititems(print_attrs)
        
def read_hdf5_dataset(file_path, dataset_name):
    with h5py.File(file_path, 'r') as file:
        if dataset_name in file:
            data = file[dataset_name][:]
            return data
        else:
            raise KeyError(f"Dataset '{dataset_name}' not found in the file.")
            
file_path = '/home/althaf/Documents/intern/TopoLoss/Code/TDFPython/finalweek/cremi_dataset.hdf'

# List contents of the HDF5 file
list_hdf5_contents(file_path)


# In[152]:


dataset_name = 'volumes/raw'

# Read data from the specified dataset
try:
    data = read_hdf5_dataset(file_path, dataset_name)
    print(f"Data shape: {data.shape}")
    #print(data)
except KeyError as e:
    print(e)

data = data/255.0 #normalized


# In[153]:


lh = data[0]
gt = lh > 0.5
print(lh,"\n",gt)


# ### Example

# In[116]:


def show_images(likelihood, groundtruth):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(likelihood, cmap='gray')
    plt.title('Likelihood')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(groundtruth, cmap='gray')
    plt.title('Groundtruth')
    plt.axis('off')
    plt.show()
show_images(lh, gt)


# ### Storing all the cropped images in a common folder

# result_folder = '/home/althaf/Documents/intern/finalweek/result/'

# for i in range(len(data)):
#     likelihood = data[i]
#     groundtruth = likelihood > 0.5
#     
#     plt.imsave(f'{result_folder}likelihood_cropped_{i+1}.png', likelihood, cmap='gray')
#     plt.imsave(f'{result_folder}groundtruth_cropped_{i+1}.png', groundtruth, cmap='gray')

# In[118]:


cremi = data
# data = cremi


# In[119]:


data = data[0:20]


# ## Training Loop for original Network

# In[148]:


num_epochs = 10
patch_size = 65
num_patches_per_epoch = 100  # Number of patches sampled in each epoch
batch_size = 25

# Initialize the list to store metrics for each epoch
metrics = []

for epoch in range(num_epochs):
    
    model.train()
    running_loss = 0.0
    total_pixel_accuracy = 0.0
    total_betti_error = 0.0
    total_ari = 0.0
    total_voi = 0.0
    
    for i in range(len(data)):
        
        likelihood = data[i]
        groundtruth = likelihood > 0.5
        
        # Sample patches from the likelihood and groundtruth arrays
        likelihood_patches = random_patch_sampling(likelihood, patch_size, num_patches_per_epoch)
        groundtruth_patches = random_patch_sampling(groundtruth, patch_size, num_patches_per_epoch)

        # Convert to tensors and create batches
        likelihood_patches = torch.tensor(likelihood_patches, dtype=torch.float32).unsqueeze(1)
        groundtruth_patches = torch.tensor(groundtruth_patches, dtype=torch.long)

        num_batches = len(likelihood_patches) // batch_size

        for b in range(num_batches):
            optimizer.zero_grad()

            # Get the current batch
            inputs = likelihood_patches[b * batch_size:(b + 1) * batch_size]
            labels = groundtruth_patches[b * batch_size:(b + 1) * batch_size]

            # Forward pass
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)  # Convert logits to probabilities
            labels1 = compute_probabilities_labels(labels)

            ce_loss = criterion(outputs, labels1)
            #print(labels1, labels.shape,"\n", labels, labels.shape)
            
            # Topological loss calculation
            inputs_np = inputs.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            topo_loss = 0.0
            #print(inputs_np.shape,labels_np.shape)

            for j in range(inputs_np.shape[0]):
                lh, bcp_lh, dcp_lh = compute_persistence_2DImg_1DHom(inputs_np[j, 0])
                gt, bcp_gt, dcp_gt = compute_persistence_2DImg_1DHom(labels_np[j])

                total_betti_error += calculate_betti_error(lh, gt)           
                topo_loss += compute_topological_loss(lh, gt)
                print(inputs_np.shape,"##########################################################3")
            topo_loss /= inputs_np.shape[0]
            total_betti_error /= inputs_np.shape[0]
            

            # Combined loss
            loss = ce_loss + topo_loss
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            total_pixel_accuracy += calculate_pixel_accuracy(outputs, labels1)
            #total_ari += calculate_ari_manual(outputs, labels1)
            total_voi += calculate_voi(outputs, labels1)
            


    avg_pixel_accuracy = total_pixel_accuracy / (num_batches*len(data))
    avg_betti_error = total_betti_error / (num_batches*len(data))
    #avg_ari = total_ari / (num_batches*len(data))
    avg_voi = total_voi / (num_batches*len(data))

    epoch_metrics = [epoch + 1, running_loss / (num_batches*len(data)), avg_pixel_accuracy, avg_betti_error, avg_voi]
    metrics.append(epoch_metrics)

    print(f"Epoch {epoch + 1}, Loss: {running_loss /(num_batches*len(data)):.4f}, Pixel Accuracy: {avg_pixel_accuracy:.4f}, Betti Error: {avg_betti_error:.4f}, VOI: {avg_voi:.4f}")


# In[149]:


metrics_df = pd.DataFrame(metrics, columns=['Epoch', 'Loss', 'Pixel Accuracy', 'Betti Error', 'VOI'])

print(metrics_df)


# # _________________________________________________________________
# 
# ## Finetuning 
# 
# # with the help of a pretrained model (VGG19)
# 

# In[137]:


class FineTunedVGG19(nn.Module):
    def __init__(self, num_classes=2):
        super(FineTunedVGG19, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True)
        
        # Modify the first convolutional layer to accept 1-channel input
        self.vgg19.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        
        # Unfreeze the last 2-3 convolutional layers
        for param in self.vgg19.features[:-10].parameters():
            param.requires_grad = False
        
        # Use layers from TopologyPreservingCNN
        self.custom_conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.custom_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Adjust the size according to the debug shapes
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)  # Adjust according to actual shape
        self.fc2 = nn.Linear(1024, num_classes)  # Binary classification
        
    def forward(self, x):
        # Pass through VGG19 up to the last few convolutional layers
        x = self.vgg19.features[:-10](x)
        print(f"Shape after VGG19: {x.shape}")
        
        # Pass through custom layers
        x = F.relu(self.custom_conv1(x))
        print(f"Shape after custom_conv1: {x.shape}")
        x = self.pool1(x)
        print(f"Shape after pool1: {x.shape}")
        x = F.relu(self.custom_conv2(x))
        print(f"Shape after custom_conv2: {x.shape}")
        x = self.pool2(x)
        print(f"Shape after pool2: {x.shape}")
        
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 128 * 4 * 4)  # Adjust according to actual shape
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


# In[138]:


# Instantiate the model
modelvgg = FineTunedVGG19()
print(modelvgg)

# Optimizer and loss function
optimizervgg = optim.Adam(modelvgg.parameters(), lr=0.001)
criterionvgg = nn.CrossEntropyLoss()


# ## training loop for the modified VGG19 model

# In[140]:


# Initialize the list to store metrics for each epoch
metrics = []

for epoch in range(num_epochs):
    
    model.train()
    running_loss = 0.0
    total_pixel_accuracy = 0.0
    total_betti_error = 0.0
    total_ari = 0.0
    total_voi = 0.0
    
    for i in range(len(data)):
        
        likelihood = data[i]
        groundtruth = likelihood > 0.5
        
        # Sample patches from the likelihood and groundtruth arrays
        likelihood_patches = random_patch_sampling(likelihood, patch_size, num_patches_per_epoch)
        groundtruth_patches = random_patch_sampling(groundtruth, patch_size, num_patches_per_epoch)

        # Convert to tensors and create batches
        likelihood_patches = torch.tensor(likelihood_patches, dtype=torch.float32).unsqueeze(1)
        groundtruth_patches = torch.tensor(groundtruth_patches, dtype=torch.long)

        num_batches = len(likelihood_patches) // batch_size

        for b in range(num_batches):
            optimizer.zero_grad()

            # Get the current batch
            inputs = likelihood_patches[b * batch_size:(b + 1) * batch_size]
            labels = groundtruth_patches[b * batch_size:(b + 1) * batch_size]

            # Forward pass
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)  # Convert logits to probabilities
            labels1 = compute_probabilities_labels(labels)

            ce_loss = criterion(outputs, labels1)
            #print(labels1, labels.shape,"\n", labels, labels.shape)
            
            # Topological loss calculation
            inputs_np = inputs.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            topo_loss = 0.0
            #print(inputs_np.shape,labels_np.shape)

            for j in range(inputs_np.shape[0]):
                lh, bcp_lh, dcp_lh = compute_persistence_2DImg_1DHom(inputs_np[j, 0])
                gt, bcp_gt, dcp_gt = compute_persistence_2DImg_1DHom(labels_np[j])

                total_betti_error += calculate_betti_error(lh, gt)           
                topo_loss += compute_topological_loss(lh, gt)
                print(inputs_np.shape,"##########################################################3")
            topo_loss /= inputs_np.shape[0]
            total_betti_error /= inputs_np.shape[0]
            

            # Combined loss
            loss = ce_loss + topo_loss
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            total_pixel_accuracy += calculate_pixel_accuracy(outputs, labels1)
            #total_ari += calculate_ari_manual(outputs, labels1)
            total_voi += calculate_voi(outputs, labels1)
            


    avg_pixel_accuracy = total_pixel_accuracy / (num_batches*len(data))
    avg_betti_error = total_betti_error / (num_batches*len(data))
    #avg_ari = total_ari / (num_batches*len(data))
    avg_voi = total_voi / (num_batches*len(data))

    epoch_metrics = [epoch + 1, running_loss / (num_batches * len(data)), avg_pixel_accuracy, avg_betti_error, avg_voi]
    metrics.append(epoch_metrics)

    print(f"Epoch {epoch + 1}, Loss: {running_loss /(num_batches*len(data)):.4f}, Pixel Accuracy: {avg_pixel_accuracy:.4f}, Betti Error: {avg_betti_error:.4f}, VOI: {avg_voi:.4f}")


# In[141]:


metricsvgg_df = pd.DataFrame(metrics, columns=['Epoch', 'Loss', 'Pixel Accuracy', 'Betti Error', 'VOI'])

print(metricsvgg_df)


# # __________________________________________________________________
# 
# # Improved CNN model with batch normalization & Dropout layers
# 

# In[130]:


#improved
class ImprovedTopologyPreservingCNN(nn.Module):
    def __init__(self):
        super(ImprovedTopologyPreservingCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)  # Adjusted for the new size
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 2)  # Assuming binary classification for segmentation

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        x = x.view(-1, 512 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# In[132]:


# Instantiate the model
modelh = ImprovedTopologyPreservingCNN()
print(modelh)

# Optimizer and loss function
optimizerh = optim.Adam(modelh.parameters(), lr=0.001)
criterionh = nn.CrossEntropyLoss()


# ## training loop for improved CNN model

# In[146]:


num_epochs = 10
patch_size = 65
num_patches_per_epoch = 100  # Number of patches sampled in each epoch
batch_size = 25

# Initialize the list to store metrics for each epoch
metrics = []

for epoch in range(num_epochs):
    
    modelh.train()
    running_loss = 0.0
    total_pixel_accuracy = 0.0
    total_betti_error = 0.0
    total_ari = 0.0
    total_voi = 0.0
    
    for i in range(len(data)):
        
        likelihood = data[i]
        groundtruth = likelihood > 0.5
        
        # Sample patches from the likelihood and groundtruth arrays
        likelihood_patches = random_patch_sampling(likelihood, patch_size, num_patches_per_epoch)
        groundtruth_patches = random_patch_sampling(groundtruth, patch_size, num_patches_per_epoch)

        # Convert to tensors and create batches
        likelihood_patches = torch.tensor(likelihood_patches, dtype=torch.float32).unsqueeze(1)
        groundtruth_patches = torch.tensor(groundtruth_patches, dtype=torch.long)

        num_batches = len(likelihood_patches) // batch_size

        for b in range(num_batches):
            optimizerh.zero_grad()

            # Get the current batch
            inputs = likelihood_patches[b * batch_size:(b + 1) * batch_size]
            labels = groundtruth_patches[b * batch_size:(b + 1) * batch_size]

            # Forward pass
            outputs = modelh(inputs)
            outputs = F.softmax(outputs, dim=1)  # Convert logits to probabilities
            labels1 = compute_probabilities_labels(labels)

            ce_loss = criterionh(outputs, labels1)
            #print(labels1, labels.shape,"\n", labels, labels.shape)
            
            # Topological loss calculation
            inputs_np = inputs.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            topo_loss = 0.0
            #print(inputs_np.shape,labels_np.shape)

            for j in range(inputs_np.shape[0]):
                lh, bcp_lh, dcp_lh = compute_persistence_2DImg_1DHom(inputs_np[j, 0])
                gt, bcp_gt, dcp_gt = compute_persistence_2DImg_1DHom(labels_np[j])

                total_betti_error += calculate_betti_error(lh, gt)           
                topo_loss += compute_topological_loss(lh, gt)
                print(inputs_np.shape,"##########################################################3")
            topo_loss /= inputs_np.shape[0]
            total_betti_error /= inputs_np.shape[0]
            

            # Combined loss
            loss = ce_loss + topo_loss
            loss.backward()

            optimizerh.step()

            running_loss += loss.item()

            total_pixel_accuracy += calculate_pixel_accuracy(outputs, labels1)
            #total_ari += calculate_ari_manual(outputs, labels1)
            total_voi += calculate_voi(outputs, labels1)
            


    avg_pixel_accuracy = total_pixel_accuracy / (num_batches*len(data))
    avg_betti_error = total_betti_error / (num_batches*len(data))
    #avg_ari = total_ari / (num_batches*len(data))
    avg_voi = total_voi / (num_batches*len(data))

    epoch_metrics = [epoch + 1, running_loss / (num_batches*len(data)), avg_pixel_accuracy, avg_betti_error, avg_voi]
    metrics.append(epoch_metrics)

    print(f"Epoch {epoch + 1}, Loss: {running_loss /(num_batches*len(data)):.4f}, Pixel Accuracy: {avg_pixel_accuracy:.4f}, Betti Error: {avg_betti_error:.4f}, VOI: {avg_voi:.4f}")


# In[147]:


metricsh_df = pd.DataFrame(metrics, columns=['Epoch', 'Loss', 'Pixel Accuracy', 'Betti Error', 'VOI'])

print(metricsh_df)


# # _________________________________________________________________
# 
# # No TopoLoss model
# 
# ## neglecting the Topo_loss and only working on the hypertuned model

# In[144]:


num_epochs = 10
patch_size = 65
num_patches_per_epoch = 100  # Number of patches sampled in each epoch
batch_size = 25

# Initialize the list to store metrics for each epoch
metrics = []

for epoch in range(num_epochs):
    
    modelh.train()
    running_loss = 0.0
    total_pixel_accuracy = 0.0
    total_betti_error = 0.0
    total_ari = 0.0
    total_voi = 0.0
    
    for i in range(len(data)):
        
        likelihood = data[i]
        groundtruth = likelihood > 0.5
        
        # Sample patches from the likelihood and groundtruth arrays
        likelihood_patches = random_patch_sampling(likelihood, patch_size, num_patches_per_epoch)
        groundtruth_patches = random_patch_sampling(groundtruth, patch_size, num_patches_per_epoch)

        # Convert to tensors and create batches
        likelihood_patches = torch.tensor(likelihood_patches, dtype=torch.float32).unsqueeze(1)
        groundtruth_patches = torch.tensor(groundtruth_patches, dtype=torch.long)

        num_batches = len(likelihood_patches) // batch_size

        for b in range(num_batches):
            optimizerh.zero_grad()

            # Get the current batch
            inputs = likelihood_patches[b * batch_size:(b + 1) * batch_size]
            labels = groundtruth_patches[b * batch_size:(b + 1) * batch_size]

            # Forward pass
            outputs = modelh(inputs)
            labels1 = compute_probabilities_labels(labels)

            loss = criterionh(outputs, labels1)

            loss.backward()
            optimizerh.step()

            running_loss += loss.item()

            total_pixel_accuracy += calculate_pixel_accuracy(outputs, labels1)
            total_voi += calculate_voi(outputs, labels1)
            

    avg_pixel_accuracy = total_pixel_accuracy / (num_batches*len(data))
    avg_betti_error = total_betti_error / (num_batches*len(data))
    #avg_ari = total_ari / (num_batches*len(data))
    avg_voi = total_voi / (num_batches*len(data))

    epoch_metrics = [epoch + 1, running_loss / (num_batches*len(data)), avg_pixel_accuracy, avg_betti_error, avg_voi]
    metrics.append(epoch_metrics)

    print(f"Epoch {epoch + 1}, Loss: {running_loss /(num_batches*len(data)):.4f}, Pixel Accuracy: {avg_pixel_accuracy:.4f}, Betti Error: {avg_betti_error:.4f}, VOI: {avg_voi:.4f}")


# In[145]:


metricsh_without_topo_df = pd.DataFrame(metrics, columns=['Epoch', 'Loss', 'Pixel Accuracy', 'Betti Error', 'VOI'])

print(metricsh_without_topo_df)


# In[ ]:




