# FoodEVPred
A Sequence-Based Computational Framework for Predicting Food-Derived Extracellular Vesicle Cargo Proteins.

- FoodEVPred is the first sequence-based machine learning tool for predicting whether a food-derived protein is extracellular vesicle (EV) cargo.
- The model is trained on a curated dataset of 11,731 protein sequences, refined to 6,356 non-redundant sequences after preprocessing.
- Proteins are classified into three categories: Milk EV, Plant EV, and Non-EV.
- Sequences are represented using embeddings from the ProtT5 pre-trained protein language model, capturing functional, structural, and stability-related context.
- FoodEVPred employs a two-tier stacked ensemble: Logistic Regression, SVM, and MLP as base classifiers, with XGBoost as the meta-learner.
- The model achieves an overall accuracy of 88.95 ± 1.26%, specificity of 94.24 ± 0.67%, and AUC of 97.32 ± 0.37%.
- FoodEVPred is deployed as a freely accessible web server for real-time and batch-mode predictions, supporting the prioritization of food-derived EV cargo proteins explored as carriers for doxorubicin prodrugs, peptide therapeutics, and antibiotics.
