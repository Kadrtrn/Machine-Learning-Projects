import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)
warnings.filterwarnings('ignore',category=FutureWarning)



df=pd.read_csv('./USArrests.csv',index_col=0)


from scipy.cluster.hierarchy import linkage
#complete ve average sık kullanıan iki farklı yöntem
hc_complete=linkage(df,'complete')
hc_average=linkage(df,'average')

from scipy.cluster.hierarchy import dendrogram 


plt.figure(figsize=(10,5))
plt.title('Hiyerarşik Kümeleme Dendogramı')
plt.xlabel('Gözlem Birimleri')
plt.ylabel('Uzaklıklar')
dendrogram(hc_complete,
          leaf_font_size=15);

#Burada istediğimiz yerden  bu grafiği bölerek kümeleyebilriz

#Biraz daha toplullaştırmış şekilde grafiği görelim

"""
lastp son p adet göster demek p =küme sayısı 
show_contracted=True kümeleme yapıldığında kaç tane küme olduğunu getirir
"""

plt.figure(figsize=(10,5))
plt.title('Hiyerarşik Kümeleme Dendogramı')
plt.xlabel('Gözlem Birimleri')
plt.ylabel('Uzaklıklar')
dendrogram(hc_complete,
           truncate_mode='lastp',
           p=4,
           show_contracted=True,
          leaf_font_size=15);

#Buraya kadar olanlar complete yöntemi içindi  aynısı average içinde yapılabilir








