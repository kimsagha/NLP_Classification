# SemanticEvolution
## ML Code Test - NLP Problem

### Project Process in Steps
1. Initialise git repository with source files
2. Create Pycharm project with conda-environment using Python 3.9 as the Python interpreter
3. Install the following packages using pip: pandas, imblearn
   UPDATE LIST OF PACKAGES
4. Read in data
5. Pre-process data
6. Train prediction model
5. Evaluate model-performance

### Challenges
1. Fixing class-imbalance (method)
   - Before:
      - Label counts: 0:10520, 1: 3695
      - F1 score: 0,9765
      - Confusion matrix: 1163(TN), 26(FP), 11(FN), 379(TP)
      - Precision: ~ 0.936, TP/(TP+FP)
      - Recall: ~0,972, TP/(TP+FN)
   - After:
      - Label counts: 0:10217, 1:10217
      - F1 score: 0,9651
      - Confusion matrix: 1133(TN), 14(FP), 41(FN), 391(TP)
      - Precision: ~ 0.965, TP/(TP+FP)
      - Recall: ~0,905, TP/(TP+FN)
   - Conclusion: recall-ratio decreased because the generation of so many
     new (non-existent) data points with the label 1 caused the model
     to overfit to that class. The SMOTETomek method makes new data points
     plausible in feature space but evidently, they are still not accurate enough.
     However, the precision-ratio increased slightly. Maybe vectorizing the text-column
     made the feature-space too high-dimensional - random over-fitting could be explored
     as an alternate option.
2. Vectorizing text-feature and combining it with other numerical (discrete and continuous) features
3. Deciding how much pre-processing to do