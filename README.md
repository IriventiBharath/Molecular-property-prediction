# Regression Chemical Language Models for Lipophilicity Prediction

This project explores the implementation and evaluation of different regression Chemical Language Models for predicting lipophilicity using the Lipophilicity dataset from the MoleculeNet benchmark. The study uses neural networks and involves fine-tuning pre-trained models, incorporating external data, and comparing various data selection and parameter-efficient fine-tuning strategies. 

#This project was a part of my course Neural Networks: Theory & Implementation by Dietrich Klakow.

## Tasks

The project is divided into three main tasks:

### Task 1: Fine-tuning and Unsupervised Fine-tuning

1.  **Fine-tuning a pre-trained model:** A pre-trained model is fine-tuned on the Lipophilicity dataset to establish a baseline for lipophilicity prediction.
2.  **Unsupervised fine-tuning:** Unsupervised fine-tuning is performed on the base model to enhance its performance. 

### Task 2: Enhancing Performance with External Data

This task focuses on improving model performance by incorporating external data.

1.  **Influence Score Calculation:** Influence functions are used to select a subset of an external dataset. 
2.  **Lissa Approximation:** The LISSA approximation method is employed to address computational challenges.

### Task 3: Data Selection and Parameter-Efficient Fine-Tuning

This task involves implementing and comparing different strategies for data selection and parameter-efficient fine-tuning. 
1.  **Data Selection:** Monte Carlo dropout is used to select the most valuable samples.
2.  **Parameter-Efficient Fine-Tuning (PEFT):** Three PEFT methods are investigated:
   * BitFit (Bias Terms Fine-Tuning)
   * LORA (Low-Rank Adaptation) 
   * iA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) 

## Results

The report includes results and analysis of each task, comparing the performance of different models and techniques. 

## References

The report cites relevant works and datasets, including the MoleculeNet benchmark and research papers on influence functions and parameter-efficient fine-tuning methods. 

## Authors

* Samuele Serri
* Bharath Vasishta Iriventi
* Omid Nezhadbahadori 
