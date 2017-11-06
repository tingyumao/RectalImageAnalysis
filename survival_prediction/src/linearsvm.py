import matplotlib.pyplot as plt
import numpy as np

## Linear SVM
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix 

from sklearn.metrics import roc_curve, auc

def linearsvm(features_nonan_np, feature_ids, labels, tmask, vmask, num_train=120, num_data=165):
    
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

    model = SVC(kernel='linear')
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
    #scores_norm = np.dot(features_norm, weights_norm)
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

    model = SVC(kernel='linear')
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
    
    
def k_fold_linearsvm(k, features_nonan_np, feature_ids, labels, num_train=137, num_data=165):
    """
    Cross validation: Repeated random sub-sampling validation

    """
    
    
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
    num_val = num_data - num_train
    
    print('num_training: {}'.format(num_train)) # 137
    print('num_validation: {}'.format(num_val)) # 28

    
    feats_weight = []
    for i in range(k):
        
        #b1 = i*(num_train-18)//k
        #b2 = i*(num_val-10)//k + 108
        b1 = np.random.randint(0,108-18)
        b2 = np.random.randint(108,num_data-10)
        # class1: class2 = 108: 57 = 1.8: 1
        vmask = [i+b1 for i in range(18)] + [i+b2 for i in range(10)]
        tmask = [ i for i in range(num_data) if i not in vmask]
    
        train_matrix = features_norm[tmask,:]
        train_labels = labels[tmask]
        valid_matrix = features_norm[vmask,:]
        valid_labels = labels[vmask]
        
        model = SVC(kernel='linear')
        model.fit(train_matrix,train_labels)

        # Train result
        result_train = model.predict(train_matrix)
        train_score = model.score(train_matrix, train_labels)
        #print(confusion_matrix(train_labels,result_train))
        #print('train score: ', train_score)

        # Valid result
        result_valid = model.predict(valid_matrix)
        valid_score = model.score(valid_matrix, valid_labels)
        #print(confusion_matrix(valid_labels,result_valid))
        #print('valid score: ', valid_score)
        
        # Plot weights(num_features,) in Linear SVM
        weights = np.squeeze(model.coef_)
        #plt.plot(weights)
        feats_weight.append(weights)
        
    # retrain with 12 selected features
    feats_np = np.asarray(feats_weight)
    feats_mean = np.mean(feats_np, axis=0)
    wid = np.argsort(-1*abs(feats_mean))[:12]
    
    features_original = features_norm
    features_select = np.squeeze(features_original[:, wid])
    feature_select_ids = [feature_ids[i] for i in wid]
    print("selected features: ")
    print(feature_select_ids)
    
    train_scores = []
    valid_scores = []
    feats_coefs = []
    for i in range(k):
        # retrain with a random train/validation set
        b1 = np.random.randint(0,108-18)
        b2 = np.random.randint(108,num_data-10)
        # class1: class2 = 108: 57 = 1.8: 1
        vmask = [i+b1 for i in range(18)] + [i+b2 for i in range(10)]
        tmask = [ i for i in range(num_data) if i not in vmask]

        train_matrix = features_select[tmask,:]
        train_labels = labels[tmask]
        valid_matrix = features_select[vmask,:]
        valid_labels = labels[vmask]

        model = SVC(kernel='linear')
        model.fit(train_matrix,train_labels)
        feats_coefs.append(model.coef_)

        # Train result
        result_train = model.predict(train_matrix)
        train_score = model.score(train_matrix, train_labels)
        train_scores.append(train_score)
        #print(confusion_matrix(train_labels,result_train))
        #print('train score: ', train_score)

        # Valid result
        result_valid = model.predict(valid_matrix)
        valid_score = model.score(valid_matrix, valid_labels)
        valid_scores.append(valid_score)
        #print(confusion_matrix(valid_labels,result_valid))
        #print('valid score: ', valid_score)
    
    # ROC curve
    feats_coef = np.mean(feats_coefs, axis=0)[0]
    
    
    #result_train = model.predict(train_matrix)
    train_score = np.dot(train_matrix, feats_coef)
    result_train = (train_score>0).astype("int")
    train_score = np.mean(result_train==train_labels)
    print(confusion_matrix(train_labels,result_train))
    print('train score: ', train_score)
    
    #result_valid = model.predict(valid_matrix)
    valid_score = np.dot(valid_matrix, feats_coef)
    result_valid = (valid_score>0).astype("int")
    valid_score = np.mean(result_valid==valid_labels)
    print(confusion_matrix(valid_labels,result_valid))
    print('valid score: ', valid_score)
    
    print("*"*40)
    print("average train score: ", np.mean(train_scores))
    print("average valid score: ", np.mean(valid_scores))
    
    y_scores = np.dot(features_select, feats_coef)
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
    
    return feats_weight
        
    
    
    
    