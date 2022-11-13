import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)
warnings.filterwarnings('ignore',category=FutureWarning)

#index_col=0 veri setinin 1. değişkenini index yapar 
df=pd.read_csv('./USArrests.csv',index_col=0)


print(df.head())

#Eksik gözlem var mı kontrol edelim 

print(df.isnull().sum())


print(df.info())

print(df.describe().T)

#n_clusters küme sayısı 
kmeans=KMeans(n_clusters=4)

#fit etme işlemi

k_fit=kmeans.fit(df)

#Kaç küme olduğunu görelim
print(k_fit.n_clusters)

#Küme merkezleri 

print(k_fit.cluster_centers_)


#Veri seti içindeki gözlemlerin hangi kümelere (cluster lara )ait olduğunu görme

print(k_fit.labels_)



#KÜMELERİN GÖRSELLEŞTİRİLMESİ

#VİDEOYA BAK !!!



#Optimum Küme Sayısının Belirlenmesi


"""
ELBOW YÖNTEMİ
"""


ssd=[]

K=range(1,30)

for k in K:
    kmeans=KMeans(n_clusters=k).fit(df)
    #Her bir k için uaklıların hesaplarını ssd ye eklemek 
    ssd.append(kmeans.inertia_)

plt.plot(K,ssd,"bx-")
plt.xlabel('Farklı k değerlerine karşılık uzaklık artık toplamları')
plt.title('Optimum küme sayısı için elbow yöntemi')
plt.show()
#Burada asıl amaç dirsek denilen yani kırılımın en sert olduğu noktadan sonraya karar vermek 
#Grafiğe bakarak bu veri seti için 2 ve 3 küme gayet mmantıklı bir seçim gibi duruyor 



"""
ELBOW YÖNTEMİİNE BİR ALTARNATİF
"""
from yellowbrick.cluster import KElbowVisualizer


kmeans=KMeans()
#k=(2,20) 2 ve 20 arasındaki k lara bak demek 
visu=KElbowVisualizer(kmeans, k=(2,20))
visu.fit(df)
visu.poof()

#öneriyide grafiğin üzerine işaretlemiş oldu



#Final k-means ---elbow yöntemine göre belirlediğimiz k yı kullanarak oluşturacagız

kmeans=KMeans(n_clusters=4).fit(df)
kmeans

#Sonuçlarla küme numaralarını bir araya getirecek bir dataframe yapalım

kumeler=kmeans.labels_

final=pd.DataFrame({'Eyaletler':df.index,'Kümeler':kumeler})

print(final)

#Finali ana veri setine ekleyelim

df['Kume_No']=kumeler

print(df)
































