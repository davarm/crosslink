# CrossLink
## A Nerual Network to Predict Disulfide Connectivity in Peptides Using Chemical Shifts
## Background
Disulfide rich peptides are highly desireable drug leads; either directly as candidates due to their exquisite potency and selectivity or their ability to be used as drug scaffolds with their highly rigid and stable structures. The disulfide bonds in these peptides act like crossbeams in a bridge, linkng the backbone resulting in highly constrained and stable conformations. The disulfides are crucial to both the structure and function of the peptides. Therefore it is essential to know the native connectivity framework (what individual cysteines oxidise with each other).

Finding the native connectivity is a factorial problem. A peptide with 3 disulfides has 15 theoretical frameworks, and 4 and 5 disulfides result in 105 and 945 possible frameworks respectively. Currently there are no definitive computational or experimental methods to resolve the connectivity framework for native peptides.

We recently showed that disulfides like to adopt specific shapes (called configurations) based on their five dihedral angles, and that the configuration can be predicted based on chemical shift inputs (https://github.com/davarm/DISH_prediction). Expanding on this concept, CrossLink is multi-stage machine learning process that incorporates both support vector machines (SVMs) and neural networks to predict the connectivity state of individual cysteine isomers. The hypothesis that was because disulfide bonds adopt specific shapes, if information regarding the configuration of individaual cysteines could be predicted then individual bonds could subsequently be predicted (a simple way of considering it is trying to match two puzzle pieces). The program achieved a baseline accuracy of 85% and >95% for high probability predictions. CrossLink is distinct to other computational methods out there in that it uses chemical shifts and structural inputs and that it returns both True and False predictions, allowing an elimination process to resolve the framework. 

## Method
The workflow of the program is :
  - Start with a peptide sequence, with no knowledge of the connectivity framework
  - Using SVMs in scikit-learn, predict the X3 angle for each individual cysteine based on chemical shifts. The X3 angle can either be      +90 or -90 degrees
  - Generate all possible pairings (isomers) of cysteine residues. Pairings where the X3 angles do not match are removed (as this is a         shared bond, therefore should always be the same)
  - Input each isomer into the neural network, using chemical shift, configuration and structural features as inputs. The neural network     will then make a prediction of the connectivity state as well as a probability for each individual isomer.
  - These predictions can be correlated with experimental data such as NOESY cross peaks. 


![Alt text](./images/method.png)


# Use
- The suppor vector machine was developed in scikit learn
- The neural network was developed with Theano using Keras frontend

### Generating Inputs (user directory)
(In this example we will use peptide PDB 2n8e)
The inputs uses adjusted chemical shifts as well as predicted backbone dihedral angles from the Talos-N program (Shen and Bax, 2013). In peptides directory add new file, with output from Talos-N (with the predAdjCS.tab and pred.tab files).

Run: 
      python generating_crosslink_inputs.py 2n8e
      
This will produce a file called 2n8e_crosslink.csv. In this file YOU MUST MANUALLY ENTER X1 angles as either -60, 60 or 180. Save the file

### X3 and Connectivity Prediction
Run:connectivity_prediction.py file 2n8e
This script will then call the modules to:
  Predict X3 Angles
  Generate all theoretical cysteine isomers and required inputs
  Run neural network and makea  prediction for connectivity for all individual isomers
