import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay


def DefineModels():
    # Spot Check Algorithms
    lr = LogisticRegression(solver='liblinear', multi_class='ovr')
    lda = LinearDiscriminantAnalysis()
    knn = KNeighborsClassifier()
    cart = DecisionTreeClassifier()
    nb = GaussianNB()
    svc = SVC(gamma='auto')

    return lr, lda, knn, cart, nb, svc

def Preprocess(dataset_original):
    # we only want to include the variables: age, sex, creatinine, LYVE1, REG1B, TFF1, and diagnosis
    attributes = ['sex', 'age', 'creatinine', 'LYVE1', 'REG1B', 'TFF1', 'diagnosis']
    dataset = pd.DataFrame(dataset_original, columns=attributes)
    
    # Encode sex: Female = 1, Male = 0
    code = {'F':1,'M':0}
    for col in dataset.select_dtypes('object'):
        dataset.loc[:,col] = dataset[col].map(code)

    diag_dict = SeparateDiagnosisData(dataset)
    
    return dataset, diag_dict

def SeparateDiagnosisData(dataset):
    # Separate diagnosis data
    # Keys - 1:Control, 2:Benign, 3:PDAC

    # Using a dict-comprehension, the 'diagnosis' value will be the key
    diag_dict = {group: data for group, data in dataset.groupby('diagnosis')}

    return diag_dict

def CompareDiagnoses(diag_dict, diag1, diag2, diag3=None):
    # Combine 2 different diagnoses to compare
    # diag1 == data for diagnosis #1, diag2 == data for diagnosis #2

    d = diag_dict[diag1].append(diag_dict[diag2])
    if (diag3 != None):
        d.append(diag_dict[diag3])
    dataComparison = d.values
    np.random.shuffle(dataComparison)

    return dataComparison

def SeparateValidation(test_size):
    #Split out validation set
    size = len(X_data)
    v_size = int(size * test_size)
    
    X_validate = X_data[:v_size-1, :]
    y_validate = y_data[:v_size-1]
    
    X_train = X_data[v_size:, :]
    y_train = y_data[v_size:]

    return X_train, y_train, X_validate, y_validate

def StandardizeData(x_data):
    sc_x = StandardScaler()
    return sc_x.fit_transform(x_data)

def TrainModel(curr_model, X_train, X_test, y_train, y_test, fold_no):
    curr_model.fit(X_train, y_train)
    predictions = curr_model.predict(X_test)
    print('Fold',str(fold_no),':',accuracy_score(y_test, predictions))

def SplitAndTrain(model, name):
    print('\nModel:', name)
    print('\nFOLD ACCURACY')
    fold_no = 1
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        TrainModel(model, X_train, X_test, y_train, y_test, fold_no)
        fold_no += 1

def EvaluatePredictions(model):
    predictions = model.predict(X_validate)
    print('\nPREDICTION EVALUATION')
    print('Accuracy:',accuracy_score(y_validate, predictions))
    print("Confusion matrix:")
    print(confusion_matrix(y_validate, predictions))
    print("Classification report:")
    print(classification_report(y_validate, predictions))

    return predictions

## Main ----------------------------------------------------------------------

dataset_original = pd.read_csv('pc_data.csv')

# # look at first 10 samples
# dataset_original.head(10)

dataset_full, diag_dict = Preprocess(dataset_original)

# Compare 2 or all 3 diagnoses
# Control(1), Benign(2), PDAC(3)
dataset = CompareDiagnoses(diag_dict, 1, 3)

x = dataset[:,0:6]
X_data = StandardizeData(x)
y_data = dataset[:, 6]

# Split out validation set
test_size = 0.2
X, y, X_validate, y_validate = SeparateValidation(test_size)

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, random_state=1, shuffle=True)

# Train models
models = []
lr, lda, knn, cart, nb, svc = DefineModels()

name = 'LR'
SplitAndTrain(lr, name)
lr_predictions = EvaluatePredictions(lr)

name = 'LDA'
SplitAndTrain(lda, name)
lda_predictions = EvaluatePredictions(lda)

name = 'KNN'
SplitAndTrain(knn, name)
knn_predictions = EvaluatePredictions(knn)

name = 'CART'
SplitAndTrain(cart, name)
cart_predictions = EvaluatePredictions(cart)

name = 'NB'
SplitAndTrain(nb, name)
nb_predictions = EvaluatePredictions(nb)

name = 'SVC'
SplitAndTrain(svc, name)
svc_predictions = EvaluatePredictions(svc)

ax = plt.gca()
# ROC from estimator
lr_disp = RocCurveDisplay.from_estimator(lr, X_validate, y_validate, ax=ax)
lda_disp = RocCurveDisplay.from_estimator(lda, X_validate, y_validate, ax=ax)
knn_disp = RocCurveDisplay.from_estimator(knn, X_validate, y_validate, ax=ax)
cart_disp = RocCurveDisplay.from_estimator(cart, X_validate, y_validate, ax=ax)
nb_disp = RocCurveDisplay.from_estimator(nb, X_validate, y_validate, ax=ax)
svc_disp = RocCurveDisplay.from_estimator(svc, X_validate, y_validate, ax=ax)

# # ROC from predictions
# lr_disp = RocCurveDisplay.from_predictions(y_validate, lr_predictions, pos_label=3, name='LR', ax=ax)
# lda_disp = RocCurveDisplay.from_predictions(y_validate, lda_predictions, pos_label=3, name='LDA', ax=ax)
# knn_disp = RocCurveDisplay.from_predictions(y_validate, knn_predictions, pos_label=3, name='KNN', ax=ax)
# cart_disp = RocCurveDisplay.from_predictions(y_validate, cart_predictions, pos_label=3, name='CART', ax=ax)
# nb_disp = RocCurveDisplay.from_predictions(y_validate, nb_predictions, pos_label=3, name='NB', ax=ax)
# svc_disp = RocCurveDisplay.from_predictions(y_validate, svc_predictions, pos_label=3, name='SVC', ax=ax)

plt.show()
