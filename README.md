# Investigating the Resilience of Audio Deepfake Detectors under Imperceptible Adversarial Attack
### Hong-Hanh Nguyen-Le (ID: 23203495)
### University College Dublin
### Course: COMP47700 - Speech And Audio

## Dataset
[Wavefake]{https://zenodo.org/records/4904579}
[ASVspoof 2019]{https://www.asvspoof.org/database}
[ASVspoof 2021]{https://www.asvspoof.org/index2021.html}

## Train deepfake detection
1. Change the dataset path in **train_models.py**
2. Change the config path in **train_models.py** which corresponds to the model that need to train in *config* folder
3. Run below:
'python train_models.py'

## Evaluate deepfake detection
1. Change the dataset path in **evaluate_models.py**
2. Change the config path in **evaluate_models.py** which corresponds to the model that need to train in *config* folder
3. Run below:
'python evaluate_models.py'

## Evaluate the adversarial examples
Run the jupytor file **evaluate_adv**  with [adversarial samples]{https://drive.google.com/file/d/1vZ8qeJYCQMSgd8rIdvD90R_F_7RLnWJc/view?usp=sharing}
