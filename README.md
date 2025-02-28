# Cauchy-Schwarz Regularizers

This is the code for the numerical results in the paper
"Cauchy-Schwarz Regularizers", S. Taner, Z. Wang, and C. Studer, ICLR, 2025.
(c) 2025 Sueda Taner, Ziyi Wang

email: taners@ethz.ch

### Important Information

 If you are using this simulator (or parts of it) for a publication, then you _must_ cite the following paper:
 
"Cauchy-Schwarz Regularizers", S. Taner, Z. Wang, and C. Studer, ICLR, 2025.

### How to use this code...

#### ...for the solution recovery results from Sections 3.1-3.3 and Appendix C

- Use folder ```solution_recovery```: 
- For recovering discrete-valued vectors:
  - For binary recovery success rate results with respect to the system dimensions as in Section 3.1, run ```binary_recovery.m```.
  - For one-sided binary and ternary recovery success rate results with respect to the linear system dimensions as in Section 3.1, run ```osb_ternary_recovery.m```.
  - For one-sided binary and ternary recovery success rate results with respect to the density as in Appendix C.4, run ```osb_ternary_vary_density.m```.
  - For recovering two-bit valued vectors as in Appendix C.5, run ```bbit_recovery.m```.
- For recovering eigenvectors as in Section 3.2, run ```eigvec_recovery.m```.
- For recovering orthogonal-column matrices as in Section 3.3, run ```orth_matrix_recovery.m```.


#### ...for the neural network quantization results from Section 3.4 

- Use folder ```quantized_nns```.
- Create your environment using ```requirements.txt```.
- Set the directory paths for ImageNet and CIFAR10 datasets in ```train_from_cfg.py```.
- Create a folder named ```pretrained_models``` and download the pretrained ResNet-20 model from [here](https://github.com/akamaster/pytorch_resnet_cifar10/tree/master/pretrained_models) into it.
- Set your training configuration in ```main_train.py```, e.g., for training ResNet-18 on ImageNet (or ResNet-20 on CIFAR-10), for binary (or ternary) regularization, and run. 


### Version history

Version 0.1: taners@ethz.ch - initial version for GitHub release.

## Acknowledgments
This project makes use of the following external code:
- [fasta-matlab](https://github.com/tomgoldstein/fasta-matlab) by T. Goldstein, accessed on 1/3/2024: used in the solution recovery results in Sections 3.1-3.3 and Appendix C.
- [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10) by I. Idelbayev, accessed on 1/12/2023: used for the pretrained ResNet-20 model in Section 3.4.

