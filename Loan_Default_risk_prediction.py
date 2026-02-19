""" LOAN DEFAULT RISK PREDICTION """
# THE CODE THE ABOUT THE CUSTOMERS WHO WAS TAKING THE LOAN FROM BANK AND WHO ARE NOT
  # importing matplotlib for visualization
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #training and test the model 
#uses matices for checking the performance,accuracy,predict the tru or false value for customer who have loan or not by confusion matrix or roc_auc,ploting with the roc curve 
from sklearn.metrics import accuracy_score,auc,precision_score,recall_score,f1_score,confusion_matrix,roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
df=pd.read_csv(r"/home/sonu-nitu/l.csv")#fetch the the data by read_csv 
df.head()
missing=df.isnull()#chechk value contains NOT-NULL or empty?
print(missing)

#totalincome
total=np.sum(df['income'])#total of whole customers income
print(total)
EMI_Income_Ratio=df['income'].value_counts(normalize=True)
print(EMI_Income_Ratio)
x=df[['income','age','credit_score','loan_amount']]
y=df['loan_status']
xtrain,xtest,ytrain,ytest=train_test_split(
    x,y,
    test_size=0.2,
    random_state=42,
    stratify=y)# ensure the both train and test have same % values  0 or 1 ?
# model
model=LogisticRegression()
model.fit(xtrain,ytrain) 

#pred,prob
ypred=model.predict(xtest)
yprob=model.predict_proba(xtest)[:,1]
y_pred_04=(yprob>0.4).astype(int)
print("Predicted probabilities:", yprob)

#matrices

print("Accuracy:", accuracy_score(ytest,y_pred_04))#accuracy of model 
print("precission",precision_score(ytest,y_pred_04))#measure the  acuuracy of positive prediction,or minmize the false positive 
#recall with predicted value above th40%
print("recall with 0.4 threshold:",recall_score(ytest,y_pred_04))

f1=f1_score(ytest,ypred)#create balance between precision and recall (find the actual positive instances)
print(f1)
roc_auc=roc_auc_score(ytest,yprob)
print("ROC AUC Score:", roc_auc)

#40% pred also used  in confusion matrix
cm=confusion_matrix(ytest,y_pred_04)#compare the predicted result against acutal data 
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6,4))
plt.imshow(cm, interpolation='bessel', cmap=plt.cm.Reds)#here bassel is the stye of coloring for predicted values, cmap =color
plt.title('Confusion_Matrix')
plt.colorbar()#give colored sidebar
tick_marks = np.arange(2)#x/y-axis ticks means label the actual or predicted conditions
                         
plt.xticks(tick_marks, ['Not Default', 'Default'], rotation=40)#rotation gives angle to xlable 
plt.yticks(tick_marks, ['Not Default', 'Default'])
plt.xlabel('Predicted_Label')
plt.ylabel('True_Label')
plt.tight_layout()#specially used for confussion matrix for adgusting the space between x/y
plt.show()  

#fpr tpr figure  measure the performance of binary classification model
fpr,tpr,thresholds=roc_curve(ytest,yprob)#measure the true positive rates and false positive rates for roc _curve by giving actual values,or prob
plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,color='Red',label='ROC_CURVE (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0, 1], color='red', linestyle='--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC_CURVE")
plt.legend(loc="lower right")
plt.show()