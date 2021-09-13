# Semantic Evolution
### ML Code Test: NLP Problem

### Project Process in Steps
1. Initialise git repository with source files
2. Create Pycharm project with conda-environment using
   Python 3.9 as the Python interpreter
3. Install the following packages using pip:
   - pandas, numpy, imblearn, tensorflow, keras
4. Read in data
5. Pre-process data
6. Create and tune the model
7. Train the final model
8. Evaluate the model's performance
9. Save model

### Code Sections
1. Lines 19-94 contain the code for preprocessing the data, 
   where the last section dealing with class-imbalance has been
   commented out as it did not significantly improve the performance
   of the final model.
   
2. Lines 98-113 contain the code for creating an SVM classifier, 
   tuning its hyperparameters (C and gamma) via a gridsearch and
   printing its performance.
   
3. Lines 116-182 contain the code for creating a neural network,
   finding the best network topology (no. of nodes per layer) via
   a random search and tuning the hyperparameters (batch-size and
   no. of epochs) via a grid search. The code will print the performance
   metrics, the tuning and training time, and lastly save the model which
   may be reloaded if needed.
   - To reload previously trained model, comment out lines 118-170, 
     and run lines 174-182. It should take about 7 seconds and the model's
     accuracy will be printed in the terminal.