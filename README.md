# Preservation-of-Topology-using-Deep-Image-Segmentation
This work deals with working on a continuous-valued loss function which is designed to enforce segmentations to maintain correct topology akin to the ground truth, ensuring consistent Betti numbers. The innovative topology-preserving loss function is differentiable, allowing seamless integration into end-to-end training of deep neural networks and then evaluating how much loss the model incurs.

# Directory
Code/TDFPython/TDFMain.py: main file for Persistence Homology
Actual link of the code of the final models are given in the readme file in ultimate folder

# Acknowledgement
This work was under the Summer Research Fellowship Program offered by Indian Academy of Sciences.<br>
Project Supervisor: Dr Amit Chattopadhyay, Assistant Professor, International Institute of Information Technology, Bengaluru.

HuXiaoling worked on to find the continuous loss function that can be incorpotated in the network. You can find it [here](https://github.com/HuXiaoling/TopoLoss). 
Inroder to find the topological loss you will be needed recompile Persistence.so file in order to run TDFMain.py wrapped as a static library.

# Methodology 
N implementation file contains only 1 sort of implementation of the Persistence Homology concept with Neural Networks.

### Various Approaches
4 total implementations have been worked on. I couldn't upload the file due to size issues.
The 4 ways are:
1. Base Model with Topological Loss
2. Fine-tuned VGG-19 model with Topological Loss
3. Enhanced CNN model with Topological Loss
4. Enhanced CNN model without Topological Loss

You can find the code [here](https://colab.research.google.com/drive/1rlalyeIvIev01O6BzPehoJ6zdYbd3gX9?usp=sharing)

## Topological Loss
The Topological Loss is incorporated in the Neural Network. You can find the a text file named "NNtopo_incorporated" which has the code for this purpose. This is just given to show which part was necessary for this work, it is already in above codes for the different models

## INCORPORATION of the loss in the network
The neural netowrk using to the loss and gradient function where the likelihood and ground truth they will be divided with a patch size of 65x65
1. We use small patches (65 × 65) instead of big patches/whole image.
2. Training using these localized topological loss can be very efficient via random patch sampling. Specifically, we do not partition the image into patches. Instead, we randomly and densely sample patches which can overlap.The patch size controls the tolerable geometric deformation. 
3. During training, even for a same patch, the diagram Dgm(f), the critical pixels, and the gradients change. 
4. At each epoch, we resample patches, reevaluate their persistence diagrams, and the loss gradients. 
5. After computing topological gradients of all sampled patches from a mini-batch, we aggregate them for backpropagation.
6. Our topological loss compliments cross-entropy loss by combating sampling bias.
7. However, for a small amount of difficult locations (blurred regions), it is much harder to learn to predict correctly. The issue is these locations only take a small portion of training pixel samples. Such disproportion cannot be changed even with more annotated training images. Topological loss essentially identifies these difficult locations during training (as critical pixels). It then forces the network to learn patterns near these locations, at the expense of overfitting and consequently slightly compromised per-pixel accuracy.
8. On the other hand, we stress that topological loss cannot succeed alone. Without cross-entropy loss, inferring topology from a completely random likelihood map is meaningless. Cross-entropy loss finds a reasonable likelihood map so that the topological loss can improve its topology.


# Referneces
[Topology Preserving Deep Image Segmentation: Xiaoling Hu, Li Fuxin, Dimitris Samaras and Chao Chen](https://proceedings.neurips.cc/paper_files/paper/2019/file/2d95666e2649fcfc6e3af75e09f5adb9-Paper.pdf)
