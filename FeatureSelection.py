import numpy as np
import pandas as pd
from sklearn import datasets
import datetime

# How below works?
'''
I will check each column's quantiles for 2 set of classes. Then I compare at what point 
the data in one class is more than other class.
e.g. class 1 has data say 
1 to 100
and another class has say from 90 to 190
now if you see that the 90% of the data in class 1 is less than class 2. This is what
I calculate but i divide this number by 2 and i will get 90/2 as 45.. max value you can get is 50
50 means min of one class is greater than or equal max of othr class and thus this column can completely
discriminate between these classes as the number starts decreasing this discrimination power will be reduced
and thus I am ranking the columns on this number for a give pair of classes and then choose best k 
columns for that class.Similarly for other classes the columns will be selected and these columns
can then be used to model the data.
 
'''

class select_features_by_importance():
    def __init__(self):
        pass
    def RankFeatures(self,dt,cls):
        clses_names = np.unique(cls)
        colnames = range(dt.shape[1])
        dt = pd.DataFrame(dt)
        clses = np.unique(cls)    
        cls = pd.DataFrame(cls)
        cls.columns = ['cls']        
        mydata = pd.merge(dt,cls,right_index=True,left_index=True)         

        qtls = np.array(range(0,100,1))/100.0        
        alldt = []
        
        for col in colnames:  
            #print (col,str(datetime.datetime.now()), '1')
            mycoldata = mydata.ix[:,[col,'cls']]
            qtls_dt = []
            #print (mycoldata)
            #print str(datetime.datetime.now()), '1'
            for i in range(len(clses)):  
                #print str(datetime.datetime.now()), '2'
                clsidata = mycoldata[mycoldata['cls']==clses[i]]
                #print (clsidata)
                #print str(datetime.datetime.now()), '3'
                #cls1_qtl = clsidata.quantile(qtls)[0]
                #print (cls1_qtl)
                qtls_dt.append(clsidata.quantile(qtls))
   
            
                #print str(datetime.datetime.now()), '4'
            #print str(datetime.datetime.now()), '5'
            #print qtls_dt
        
            for i in range(len(clses)-1):  
                cls1_qtl = qtls_dt[i]  
                for j in range(i+1,len(clses)):
                    #print str(datetime.datetime.now()), '6'
                    cls2_qtl = qtls_dt[j]
                    #reverse the class j quantiles
                    cls2_qtl=cls2_qtl.sort_index(ascending=False)
                    #print str(datetime.datetime.now()), '7'
                    #subtract quantile values of class i from class j
                    myd= np.array(cls2_qtl)[:,0] - np.array(cls1_qtl)[:,0]
                    #see where the data is bigger than 0 and less than 0
                    
                    mx = len(myd[myd > 0])
                    mn = len(myd[myd < 0])
    
                    #print (mx,mn)
                    if mx==mn==0:
                        a= 0
                    else:
                        if abs(mn-50) <= abs(mx-50 + 1):
                            a = abs(mn-50)
                        else:
                            a= abs(mx-50 + 1)
    
                    ptdt = list([clses_names[i],clses_names[j],col,a])
                    alldt.append(ptdt)
                    #print str(datetime.datetime.now()), '7'
                    #print (g)

            #print (col,str(datetime.datetime.now()), '2')
        
        alldt_df = pd.DataFrame(alldt)
        alldt_df.columns = ['cls1','cls2','col','qtl']
        alldt_df.set_index(['cls1','cls2'])
        
        #rank the data and use method first so that the first column get priority. You can change it if you want.
        # this decision is not based on anything but on preferences.
        #print ('alldt_1')
        #print (alldt_df)
        alldt_df['rnk'] = alldt_df.groupby(['cls1','cls2'])['qtl'].rank(ascending=False,method='first')
        #print ('alldt')
        #print (alldt_df)
        return (alldt_df)
        # Now you have the ranks and now you can select top say 1 or 2 or 3 for each class and then use this data for 
        #your modeliing and feature reduction.

    def ReturnFeatureRanks(self,X,Y):
        
        return self.RankFeatures(X,Y)
        
if __name__ =='__main__':
    dgts = datasets.load_digits()
    dgts_data = dgts.images
    rownum = dgts_data.shape[0]
    dgts_data= dgts_data.reshape(rownum,64)
    dgts_data = dgts_data
    dgts_labels = dgts.target
    a = select_features_by_importance()
    alldt_df = a.ReturnFeatureRanks(dgts_data,dgts_labels)

    k = 5

    topkfeatures = np.unique(alldt_df[alldt_df['rnk'] <= k]['col'])
    print (topkfeatures)
        