import matplotlib.pyplot as plt
import numpy as np

## Linear SVM
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix 

from sklearn.metrics import roc_curve, auc

def svm(features_nonan_np, feature_ids, labels, tmask, vmask, num_train=120, num_data=165, kernel="rbf"):
    
    # Normalize data
    features_mean = np.mean(features_nonan_np, axis=0)
    features_std = np.std(features_nonan_np, axis=0)
    features_std = np.maximum(features_std, 1e-15)
    features_norm = (features_nonan_np - features_mean)/features_std
    # Add bias term
    features_nb = np.hstack((features_norm, np.ones((features_norm.shape[0],1))))

    print('feature\'s shape: {}'.format(features_norm.shape))
    
    ## Start Linear SVM
    ## Randomly select 100 samples for training and other(46) leave for validation. 10/4
    #num_train = 120
    #num_data = 165
    #tmask = np.random.choice(num_data, size=num_train, replace=False).tolist()
    #vmask = [ i for i in range(num_data) if i not in tmask]

    print('num_training: {}'.format(len(tmask)))
    print('num_validation: {}'.format(len(vmask)))

    train_matrix = features_norm[tmask,:]
    train_labels = labels[tmask]
    valid_matrix = features_norm[vmask,:]
    valid_labels = labels[vmask]

    # pre-train
    model = SVC(kernel=kernel)
    model.fit(train_matrix,train_labels)

    # Train result
    result_train = model.predict(train_matrix)
    train_score = model.score(train_matrix, train_labels)
    print(confusion_matrix(train_labels,result_train))
    print('train score: ', train_score)

    # Valid result
    result_valid = model.predict(valid_matrix)
    valid_score = model.score(valid_matrix, valid_labels)
    print(confusion_matrix(valid_labels,result_valid))
    print('valid score: ', valid_score)
    
    # Plot weights(num_features,) in Linear SVM
    weights = np.squeeze(model.coef_)
    #plt.plot(weights)

    weights_norm = weights
    
    ## SELECT TOP K FEATURES
    K = 12
    # original scores for comparison
    #scores_nb = np.dot(features_nb, weights_nb)
    #weights = weights_nb
    #features_original = features_nb
    scores_norm = np.dot(features_norm, weights_norm)
    weights = weights_norm
    abs_weights = abs(weights_norm)
    features_original = features_norm
    # Filter out some small weights
    wid = np.argsort(abs_weights)[-K:]
    print('num of selected features: {}'.format(wid.shape[0]))
    weights_select = weights[wid]
    features_select = np.squeeze(features_original[:, wid])
    scores_select = np.dot(features_select, weights_select)
    
    feature_select_ids = [feature_ids[i] for i in wid]
    print(feature_select_ids)
    
    # retrain with 12 selected fetures
    train_matrix = features_select[tmask,:]
    train_labels = labels[tmask]
    valid_matrix = features_select[vmask,:]
    valid_labels = labels[vmask]

    print('num_train & num of features: {}'.format(train_matrix.shape))
    print('num_validation & num of features: {}'.format(valid_matrix.shape))

    # retrain
    model = SVC(kernel=kernel)
    model.fit(train_matrix,train_labels)

    
    print("*"*80)
    print("Retrain result")
    # Train result
    result_train = model.predict(train_matrix)
    train_score = model.score(train_matrix, train_labels)
    print(confusion_matrix(train_labels,result_train))
    print('train score: ', train_score)

    # Valid result
    result_valid = model.predict(valid_matrix)
    valid_score = model.score(valid_matrix, valid_labels)
    print("valid result: ")
    print(confusion_matrix(valid_labels,result_valid))
    print('valid score: ', valid_score)
    
    # ROC curve
    y_scores = np.dot(features_select, model.coef_.T)
    y_true = labels
    fpr, tpr, threshold = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    
    
    