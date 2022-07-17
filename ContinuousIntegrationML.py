# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 12:02:30 2022

@author: Şiyar Öner
"""
#libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#set random seed
seed=42

#import data
df=pd.read_csv("FermentedGrapeJuiceQuality.csv")

#train test split
y=df.pop("quality")
x_train,x_test,y_train,y_test= train_test_split(df,y,test_size=0.2, random_state=seed)


#### Creating Model
#fit
rfr=RandomForestRegressor(max_depth=2,random_state=seed)
rfr.fit(x_train,y_train)

#training and test score
score_train=rfr.score(x_train,y_train)*100
score_test=rfr.score(x_test,y_test)*100

# saving score to a file
with open("FermetedGrapeJuiceScores.txt","w") as f:
    f.write("Training varience score: %2.1f%%\n"%score_train)
    f.write("test varience score: %2.1f%%\n"%score_test)


# visualize
importances=rfr.feature_importances_
labels=df.columns
feature_df=pd.DataFrame(list(zip(labels,importances)),columns=["Feature","Importance"])
featue_df=feature_df.sort_values(by="Importance",ascending=False,)

#image format
axis_fs=18 #fontsize
title_fs=22
sns.set(style="whitegrid")

ax=sns.barplot(x="Importance",y="Feature",data=feature_df)
ax.set_xlabel("importance",fontsize=axis_fs)
ax.set_ylabel("feature",fontsize=axis_fs)
ax.set_title("Random Forest\nFeature importance",fontsize=title_fs)

plt.tight_layout()
plt.savefig("Feature_importance.png",dpi=300)
plt.close()

# plot
pred=rfr.predict(x_test)+np.random.normal(0,0.25,len(y_test))
y_jitter=y_test+np.random.normal(0,0.25,len(y_test))
res_df=pd.DataFrame(list(zip(y_jitter,pred)),columns=["True","Pred"])


ax = sns.scatterplot(x="True", y="Pred",data=res_df)
ax.set_aspect('equal')
ax.set_xlabel('True FGJ quality',fontsize = axis_fs) 
ax.set_ylabel('Predicted FGJ quality', fontsize = axis_fs)#ylabel
ax.set_title('Residuals', fontsize = title_fs)

# Make it pretty- square aspect ratio
ax.plot([1, 10], [1, 10], 'black', linewidth=1)
plt.ylim((2.5,8.5))
plt.xlim((2.5,8.5))

plt.tight_layout()
plt.savefig("residuals.png",dpi=120) 























