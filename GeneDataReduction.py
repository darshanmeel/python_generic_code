
from FeatureSelection import select_features_by_importance
import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2

from sklearn import datasets
import sklearn
from sklearn import grid_search
from sklearn import metrics
from sklearn import svm

from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier
import datetime
from sklearn.metrics import confusion_matrix, classification_report 



def fit_pred_return_efficiency(clf,train,test,traincls,testcls):
    clf.fit(train, traincls)    
    cls_pred = clf.predict(test)
    accuracy_score = metrics.accuracy_score(testcls,cls_pred)
    print classification_report(testcls,cls_pred)
    return cls_pred,accuracy_score
    
# it will return the top 10 accuracies for diff parameters
def gettopkaccuracies(crts,n=10):
    # Make a pandas data frame from list crts.
    crts_pd = pd.DataFrame(crts)
    crts_pd.columns = ['fold','params','accuracy']
    # Group based on the second column i.e. params
    prm_grpd = crts_pd.groupby('params')
    accuracy_mean = pd.DataFrame(prm_grpd.mean()['accuracy'])
    top_params_withhigh_accuracy = accuracy_mean.sort(['accuracy'],ascending=False)[0:n]
    accuracy_std = pd.DataFrame(prm_grpd.std()['accuracy']).fillna(0.0)
    top_params_withhigh_accuracy = pd.merge(top_params_withhigh_accuracy,accuracy_std,right_index=True,left_index=True)
    return top_params_withhigh_accuracy



gen = pd.read_csv('/Users/dsing001/JoinedandTransposedGenData')

# create the array of classes and declare it as string of length 10
gen_cls = np.array(gen.ix[:,12625], dtype="|S10")

# create the array of the data
gen = np.array(gen.ix[:,0:12625])

# create unique array of classes
clses =np.unique(gen_cls)

clses_names = clses
print (clses)
print (clses_names)
print (len(clses))
for i in range(4):
    print (i)


colnames = np.arange(gen.shape[1])
print (colnames)

# define k i.e. number of imp parameters to return
kvals= (4,5)

# Let us apply the adaboost with naive bayes and see how will  it work..
parameters = {'tol':[0.0001,0.001], 'C':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
clf_admn = AdaBoostClassifier(base_estimator=MultinomialNB(fit_prior=True),n_estimators=100)
clf_knn = KNeighborsClassifier(5)
clf_nb = MultinomialNB(fit_prior=True)
clf_svm = svm.SVC(C= 100,gamma=0.001,kernel='rbf')
clf_rf = RandomForestClassifier(n_estimators=500, min_samples_split=3, min_samples_leaf=2)
clf_lg = sklearn.linear_model.LogisticRegression()
clf = grid_search.GridSearchCV(clf_lg, parameters,scoring='recall')
num_fold = 2
gen_10_sets = StratifiedKFold(gen_cls,num_fold,True)
gen_10_sets = StratifiedShuffleSplit(gen_cls, n_iter=1, test_size=0.1)

clfs = [clf_admn,clf_knn,clf_nb,clf_svm,clf_rf,clf_lg]
clfs = [clf]
crts=[]
kvals = range(4,5)
fold= 0
a = select_features_by_importance()
print (num_fold)
readfromfiles = 0
for train_index, test_index in gen_10_sets:   
    fold= fold + 1


    if readfromfiles == 0:
        #gen_train, gen_test = gen_pca[train_index], gen_pca[test_index]
        gen_train, gen_test = gen[train_index], gen[test_index]
        gen_cls_train, gen_cls_test = gen_cls[train_index], gen_cls[test_index]
        
        pd.DataFrame(gen_train).to_csv('gen_train.csv')
        pd.DataFrame(gen_cls_train).to_csv('gen_cls_train.csv')
        pd.DataFrame(gen_cls_test).to_csv('gen_cls_test.csv')
        pd.DataFrame(gen_test).to_csv('gen_test.csv')
        
        print fold,str(datetime.datetime.now()), 'starts'    
        alldt_df = a.ReturnFeatureRanks(gen_train,gen_cls_train)
        #alldt_df = pd.read_csv('alldt_df.csv')
        #print alldt_df
        alldt_df.to_csv('alldt_df.csv')
    else:
        alldt_df = pd.read_csv('alldt_df.csv')
        gen_train = np.array(pd.read_csv('gen_train.csv',index_col = 0,skiprows=0))
        gen_cls_train = np.array(pd.read_csv('gen_cls_train.csv',index_col = 0,skiprows=0))
        gen_test = np.array(pd.read_csv('gen_test.csv',index_col = 0,skiprows=0))
        gen_cls_test = np.array(pd.read_csv('gen_cls_test.csv',index_col = 0,skiprows=0))

    print fold,str(datetime.datetime.now()), 'ends'

    for k in kvals:
        print 
        print ( 'for top %s features' % (str(k)) )
        print
        
        topkfeatures = np.unique(alldt_df[alldt_df['rnk'] <= k]['col'])
        print topkfeatures
        train_data = gen_train[:,topkfeatures]
        train_class = gen_cls_train
        test_data = gen_test[:,topkfeatures]
        test_class = gen_cls_test
        print (train_data.shape)
        print (train_class.shape)
        print (test_data.shape)
        print (test_class.shape)
        for i,clf in enumerate(clfs):
            modelname= str(clf)
            endstr = modelname.find('(')
            modelname= modelname[:endstr]

            print 
            print ('starting for model ', modelname)
            print
            cls_pred,accuracy =fit_pred_return_efficiency(clf,train_data,test_data,train_class,test_class)
            print ('accuracy is ' ,accuracy  )  

            parameters = i,modelname,k
            # Add it to a list
           
            crt = fold,parameters,accuracy
            # Append this list to the main list which contains details for all the loops
            crts.append(crt)
   
k = 10000
topk_params_withhigh_accuracy =gettopkaccuracies(crts,k)

print
print "Top " + str(k) + " models"
print

print topk_params_withhigh_accuracy
            

        