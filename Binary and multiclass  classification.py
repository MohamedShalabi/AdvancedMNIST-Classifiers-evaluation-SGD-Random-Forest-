
# coding: utf-8

# In[1]:


#step one : fething the dataset
from sklearn.datasets import fetch_mldata
mnist  = fetch_mldata('MNIST original')
mnist


# In[30]:


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
# to make this notebook's output stable across runs
np.random.seed(42)
# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")


# In[2]:


# step two : setup features matrix and label vector
x = mnist['data']
y= mnist['target']
x.shape


# In[3]:


y.shape


# In[4]:


# step3 take a loo on an istance
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt

imgpixel = x[36000]
imgframe=imgpixel.reshape(28,28)
plt.imshow(imgframe ,cmap = matplotlib.cm.binary , interpolation = 'nearest')
y[36000] #just to assure that features meets the target in the dataset


# In[5]:


# step four :splitting the data into train and test 
import pandas as pd 
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=10/70, random_state=42)


# In[6]:


# step five : shuffling the train data (make them random!)
import numpy as np
shuffler_tool = np.random.permutation(60000)
xtrain , ytrain = xtrain[shuffler_tool] , ytrain[shuffler_tool]
xtrain


# In[7]:


# Training a binary classifier for identifying one digit
# Step six : creat a binary target victor for number 5
train_target5 = (ytrain==5)
test_traget5 = (ytest==5)
train_target5


# In[8]:


# step7 : lets try with the first model :Stochastic Gradient Descent) SGD : GOOD FOR BINARY and big amont of data
from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier(random_state=42)
classifier.fit(xtrain,train_target5)
# check a predection for example we know that instance 36000 has a traget with 5
classifier.predict([x[36000]])


# In[9]:


# step8: evaluate the performance
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier , xtrain , train_target5 , cv =3 , scoring ='accuracy')
scores


# In[10]:


# step9: let's build a confusion matrix between true values and predicted
from sklearn.model_selection import cross_val_predict
predict_from_kfolds =cross_val_predict(classifier , xtrain , train_target5 , cv=3)
from sklearn.metrics import confusion_matrix
confusion_matrix(train_target5 , predict_from_kfolds)


# In[11]:


# step 10 : get out all scores 
from sklearn.metrics import precision_score , recall_score ,f1_score
def get_allscores(true , predict):
    precision = precision_score(true , predict)
    recall = recall_score(true , predict)
    f1 = f1_score(true , predict)
    M = print('precision:', precision ,'recall :' , recall , 'f1: ' , f1)
    return M

get_allscores(train_target5 , predict_from_kfolds)  


# In[12]:


# lets draw the recall precision curve to decide on our threshold
yscores = cross_val_predict(classifier ,xtrain, train_target5 ,cv = 3 , method ='decision_function')


# In[13]:


# step11: plotting the curve
from sklearn.metrics import precision_recall_curve
precisions , recalls , thresholds = precision_recall_curve(train_target5 ,yscores)
def plot_prec_recall():
    plt.plot(thresholds , precisions[:-1], 'b--' , label ='precision')
    plt.plot(thresholds , recalls[:-1] , 'g-' , label = 'recall')
    plt.xlabel('Thresholds')
    plt.legend(loc= 'upper left')
    plt.ylim([0 ,1])
plot_prec_recall()


# In[14]:


# step12 : selectthe required precision then set the thrshpld for it : let's say 90% precision 
newprediction = (yscores>200000)
get_allscores(train_target5 , newprediction)  


# In[15]:


#step13: using ROC curve method for checking True Positive rates and False Positive Rates
from sklearn.metrics import roc_curve
fpr , tpr , thresholds = roc_curve(train_target5 , yscores)
def plot_roc(fpr ,tpr ,  label = None ):
    plt.plot(fpr , tpr  ,  label = label)
    plt.plot([0,1],[0,1],'g--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
plot_roc(fpr ,tpr)


# In[16]:


# step14: get the area under ROC curve ; ROC score
from sklearn.metrics import roc_auc_score
roc_auc_score(train_target5 , yscores)


# In[17]:


# step15: let's use another model and compare:random forest
from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(random_state = 42)
yprobapility = cross_val_predict(classifier2 ,xtrain , train_target5 , cv=3 , method = 'predict_proba')
yscoresrandom = yprobapility[: ,1]
fprrand , tprran , thresholdsrand =roc_curve(train_target5 , yscoresrandom)
plot_roc(fprrand ,tprran ,'Radom Forest')
plt.plot(fpr, tpr , 'b:' , label = 'SGD')
plt.legend(loc = 'lower right')
plt.show()


# In[18]:


# get the random forest ROC score
from sklearn.metrics import roc_auc_score
roc_auc_score(train_target5 ,yscoresrandom)#0.992 means better than SGD


# In[19]:


# notes to keep in mid
# note1:SGD Classifier is good for big data and bbinary classifier 
# note2: theshold is very important to adjust the precision nd recall
# note3: ther are two tools for binarssten to chec the trades off between recall and precision(pr cuurve and ROC curve)
# note4: precision is how much true positive compared to all your predicted positives but recall is how big enough is our true positive compared
#     to the all real true positives


# In[20]:


# Now we will try Multiclassification
#    some important notes 
# 1:Random classifiier and Naive Bayes handle directly Multiple classes
# 2:Support vector machines and Linear classifiers handle directly Binary classifiers.
# 3: Two ways can be used for binary to make it used for multy classes(OVO /OVA)
# 4: Always when using Binary for Multy on scikit Learn it uses OVA excep Support vector M uses OVO.
# 5:OVO Rule --> Number of classes will be ((N-1)/2)*N where N is the number of label Unique Values


# In[21]:


# Step1: get the data 
# xtrain ,xtest ytrain and ytest the same like privious example
# step2: get the model to fit
from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier(random_state=42)
classifier.fit(xtrain,ytrain)
classifier.predict([x[36000]])


# In[22]:


# step3: tae alook onthe classes of the classifiers
print(classifier.classes_)


# In[23]:


# Step4: check how many classifier we have and get there scores for the selected  digit (X[36000])
multiscores =classifier.decision_function([x[36000]])
print(multiscores) #As below five has the highst score for that digit
print(np.argmax(multiscores)) #so class 5 is the highst score


# In[24]:


# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")
# Note: we can force scikit to use any strategy(OVO or OVA)
from sklearn.multiclass import OneVsOneClassifier
ovo = OneVsOneClassifier(SGDClassifier(random_state=42))

ovo.fit(xtrain , ytrain)
ovo.predict([x[36000]])


# In[25]:


# step5: let's use another model and compare:random forest
from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(random_state = 42)
classifier2.fit(xtrain , ytrain)
print(classifier2.predict([x[36000]]))
classifier2.classes_


# In[26]:


# Note: for Random Forest We get Probabilties not scores
probabilit_notScores =classifier2.predict_proba([x[36000]])
print(probabilit_notScores)
print(probabilit_notScores.argmax())#Same like SGD or any classifiers you can get the max prob 0.8


# In[27]:


# Step6: Lets apply some evaluation for each classifier
# for SGD multiclassifier
#         for accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier , xtrain , ytrain , cv =3 , scoring ='accuracy')
print(scores)
# for Random Forest accuracy
from sklearn.model_selection import cross_val_score
scores2 = cross_val_score(classifier2 , xtrain , train_target5 , cv =3 , scoring ='accuracy')
print(scores2)#below ou will see how random is more accurate


# In[28]:


# step6.1: get the confusion matrix
crossvalPredict = cross_val_predict(classifier ,xtrain , ytrain , cv = 3)
conf = confusion_matrix(ytrain , crossvalPredict)
print(conf)
# lets plot
plt.matshow(conf , cmap =plt.cm.gray)
plt.show()


# In[29]:


# step6.2: build conf _matrix with error rates
# for each raw of the conf matric we sum the all values  to know the all real valuesthen devide the prediction with all real vlaues
each_real_sum = conf.sum(axis=1 , keepdims=True)
print(each_real_sum)
# normalization the conf_matrix
norm_conf = conf/each_real_sum
print(norm_conf)
np.fill_diagonal(norm_conf , 0)
plt.matshow(norm_conf , cmap = plt.cm.gray)


# In[69]:


def digits_plt(instances ,number_perraw = 10,**options):
    size = 28
    num_raws = (len(instances)-1)//number_perraw + 1
    images = [instance.reshape(size,size) for instance in instances]
    empty_place  = []  #empt place for all images per raw 
    n_empty = num_raws * number_perraw - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for raw in range(num_raws):
        all_images_per_raw = images[raw*number_perraw : (raw+1) * number_perraw]
        empty_place.append(np.concatenate(all_images_per_raw,axis = 1))
    Big_image = np.concatenate(empty_place , axis = 0)
    plt.imshow(Big_image , cmap = matplotlib.cm.binary ,**options)
    plt.axis("off")


# In[77]:


classA ,classB = 3,5
x_aa = xtrain[(crossvalPredict== classA) & (ytrain == classA)]
x_BA = xtrain[(crossvalPredict == classA) & (ytrain ==classB)]
plt.figure(figsize=(8,8))
plt.subplot(221);digits_plt(x_aa[:30],number_perraw = 5)
plt.subplot(222);digits_plt(x_BA[:30],number_perraw = 5)

