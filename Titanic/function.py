import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt



def bar_plot(df,x,y,hue):
    sns.barplot(x = x,
            y = y,
            hue = hue,
            data = df);

def scatter_plot(df,X,Y,hue=None,style=None):
    sns.scatterplot(data=df, x=X, y=Y, hue=hue, style=style);

def hist(df,X,hue=None):
    sns.histplot(data=df, x=X, hue=hue)
    plt.show()
    
def pie_chart(df,data):
    fig = plt.gcf()
    fig.set_size_inches(6,6) 
    plt.pie(df[data].value_counts(),
    labels=pd.unique(df[data]),
    colors=sns.color_palette('muted'),
    autopct='%3.2f%%',
    textprops={'fontsize': 14})
    plt.show()


def boxplot(df,x,y=None,hue=None):
    sns.boxplot(data=df, x=x, y=y, hue=hue)
    
    
def outlier_limits(df,col,q1=0.5,q3=0.95):
    q1=df[col].quantile(q1)
    q3=df[col].quantile(q3)
    IQR=q3-q1
    lower_bound=q1-1.5*IQR
    upper_bound=q3+1.5*IQR
    print(col,'için alt sınır',lower_bound)
    print(col,'için üst sınır',upper_bound)

def outlier_detector(df,col,q1=0.5,q3=0.95):
    q1=df[col].quantile(q1)
    q3=df[col].quantile(q3)
    IQR=q3-q1
    lower_bound=q1-1.5*IQR
    upper_bound=q3+1.5*IQR
    if df[(df[col] > upper_bound) | (df[col] < lower_bound)].any(axis=None):
        return True
    else:
        return False

def outlier_clear(df,col,q1=0.5,q3=0.95):
    q1=df[col].quantile(q1)
    q3=df[col].quantile(q3)
    IQR=q3-q1
    lower_bound=q1-1.5*IQR
    upper_bound=q3+1.5*IQR
    
    df = df[~((df[col] < lower_bound) | (df[col] > upper_bound))]
    return df 