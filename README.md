# Python Project to Analyze the Wisconsin Breast Cancer Dataset

1.  The Wisconsin Breast Cancer Data Set is a popular data set on the UCI Machine Learning Repository<sup>1</sup>
2.  The data set consists of 569 subjects with 30 measured features per subject and ground truth of benign or malignant for each subject
3.  The original paper showed an accuracy of 97% using 10-fold cross-validation via the usage of a linear programming-based inductive classifier with three of the 30 features for the separating plane
4.  Exploratory analysis was performed to identify the most significant features and to identify subjects that are the largest outliers 
5.  PLSDA models were built and reviewed for a variety of features, including the usage of additional engineered features and subject removal for the investigation of outliers
6.  The best model with all subjects and with 38 features (including 8 engineered features) showed an accuracy of 97% using leave-one-out cross-validation
7.  An advantage of PLSDA is that it provides diagnostics and robust models
8.  Interesting techniques applied include:
    - The use of boxplots and overlapping subjects per feature to identify the features with the highest discrimination power
    - The use of subject-based correlation images to identify subject outliers and to quantify the homogeneity of the benign and malignant subject populations
    - The addition of engineered features to the original features, where the engineered features are combinations of the most discriminative original features
    - The use of the interactive 3D Plotly graphing tool to identify outliers among the misclassifications that are in the boundary between the benign and malignant clusters
    - The use of automated subject removal model building to explore the outliers via the generation of a large subset of models


<sup>1</sup>Street, W. Nick, William H. Wolberg, and Olvi L. Mangasarian. "Nuclear feature extraction for breast tumor diagnosis." Biomedical image processing and biomedical visualization. Vol. 1905. SPIE, 1993
