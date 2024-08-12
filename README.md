# Topology-Preservation-Using-Deep-Image-Segmentation

Introducing a continuous-valued loss function. This function is designed to enforce segmentations to maintain correct topology akin to the ground truth, ensuring consistent Betti numbers. The innovative topology-preserving loss function is differentiable, allowing seamless integration into end-to-end training of deep neural networks.

# Directory
Code/TDFPython/TDFMain.py: main file for Persistence Homology
Actual link of the code of the final models are given in the readme file in ultimate folder

# Acknowledgement
HuXiaoling worked on https://github.com/HuXiaoling/TopoLoss. You can see their repository. You will be needed recompile Persistence.so file in order to run TDFMain.py wrapped as a static library.
