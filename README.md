## EXNO-5-DS-DATA VISUALIZATION USING MATPLOT LIBRARY
# Name: SHYAM R
# REG.No:212223040200
# Aim:
To Perform Data Visualization using matplot python library for the given datas.

# EXPLANATION:
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

# Algorithm:
STEP 1:Include the necessary Library.

STEP 2:Read the given Data.

STEP 3:Apply data visualization techniques to identify the patterns of the data.

STEP 4:Apply the various data visualization tools wherever necessary.

STEP 5:Include Necessary parameters in each functions.

# Coding and Output:
Include the necessary coding and corresponding screenshots
```
import matplotlib.pyplot as plt
x1=[1,2,3]
y1=[2,4,1]
plt.plot(x1,y1,label='line1')
x2=[1,2,3]
y2=[4,1,3]
plt.plot(x2,y2,label='line2')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Two lines on same graph!')
plt.legend()
plt.show()
```
![image](https://github.com/user-attachments/assets/faf854c1-5e11-4b06-b3ff-ebf45bf48c14)

```
import matplotlib.pyplot as plt
x=[1,2,3,4,5,6]
y=[2,4,3,5,1,6]
plt.plot(x,y,color='green',linestyle='dashed',linewidth=3,marker='o',markerfacecolor='red',markersize=10)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Some cool customizations!')
plt.show()
```
![image](https://github.com/user-attachments/assets/558493b7-da34-4c2c-915d-aff04a24b39b)

```
yield_apples=[0.895,0.91,0.919,0.926,0.929,0.931]
plt.plot(yield_apples)
```
![image](https://github.com/user-attachments/assets/7fffd09f-c350-4907-8939-dd9e081a9d20)

```
years=[2010,2011,2012,2013,2014,2015]
yield_apples=[0.895,0.91,0.919,0.926,0.929,0.931]
plt.plot(years,yield_apples)
```
![image](https://github.com/user-attachments/assets/de6e7be3-6d11-4f04-9583-8f8ad5ab6ea1)

```
years=range(2000,2012)
apples=[0.895,0.91,0.919,0.926,0.929,0.931,0.934,0.936,0.937,0.9375,0.9372,0.939]
grapes=[0.926,0.941,0.930,0.923,0.918,0.908,0.907,0.904,0.901,0.898,0.9,0.896, ]
plt.plot(years, apples)
plt.plot(years, grapes)
plt.xlabel('Year')
plt.ylabel('Yield (tons per hectare)')
plt.title('Crop Yields')
plt.legend(['Apples','grapes']);
```
![image](https://github.com/user-attachments/assets/584c2909-2f5e-4c25-b0a5-b7b76d3559a1)

```
plt.figure(figsize=(14,6))
plt.plot(years,grapes,marker='o')
plt.title("Yield of grapes (tons per hectare)");
```
![image](https://github.com/user-attachments/assets/09cf342d-a56d-49dc-9c77-585669b9894b)

```
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
x=np.arange(0,10)
y=np.arange(11,21)
x
```
![image](https://github.com/user-attachments/assets/c30ed1cf-0851-4e82-aae6-f3ca91d7c335)

```
y
```
![image](https://github.com/user-attachments/assets/ace1ff04-e5f4-4b1b-bb57-1bd8546bccfd)

```
plt.scatter(x,y,c='b')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Graph in 2D')
plt.savefig('Test.png')
```
![image](https://github.com/user-attachments/assets/b2b05454-606e-4e3f-add7-a3b444d653f6)

```
y=x*x
y
```
![image](https://github.com/user-attachments/assets/946ed4ac-64a8-4323-aeb1-bf5a2e3c9412)

```
plt.plot(x,y,'y*',linestyle='dashed',linewidth=2,markersize=12)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('2d Diagram')
```
![image](https://github.com/user-attachments/assets/1ae265bd-0b5e-4dd4-8813-f67c7baa1b9d)

```
np.pi
```
![image](https://github.com/user-attachments/assets/22fe1183-5f3d-4f9b-a32a-1a2a867160c6)

```
x=np.arange(0,4*np.pi,0.1)
y=np.sin(x)
plt.title("sine wave form")
plt.plot(x,y)
plt.show()
```
![image](https://github.com/user-attachments/assets/e29ccb5a-f53a-43a4-8b35-58cca287696a)

```
import matplotlib.pyplot as plt
import numpy as np
x=[1,2,3,4,5]
y1=[10,12,14,16,18]
y2=[5,7,9,11,13]
y3=[2,4,6,8,10]
plt.fill_between(x,y1,color='red')
plt.fill_between(x,y2,color='green')
```
![image](https://github.com/user-attachments/assets/4d4f9401-05fa-49eb-a0a7-4b74a3ad9bbe)

```
import matplotlib.pyplot as plt
import numpy as np
x=[1,2,3,4,5]
y1=[10,12,14,16,18]
y2=[5,7,9,11,13]
y3=[2,4,6,8,10]
plt.fill_between(x,y1,color='blue')
plt.fill_between(x,y2,color='green')

plt.plot(x,y1,color='red')
plt.plot(x,y2,color='black')
plt.legend(['y1','y2'])
```
![image](https://github.com/user-attachments/assets/09ef4b46-e229-40ab-be21-2b4dd44167db)

```
import matplotlib.pyplot as plt
height=[10,24,36,40,5]
names=['one','two','three','four','five']
c1=['red','green']
c2=['b','g']
plt.bar(names,height,width=0.8,color=c1)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('My bar chart!')
plt.show()
```
![image](https://github.com/user-attachments/assets/30a373a0-71cc-43cc-9f8b-30652cee8283)

```
x=[2,8,10]
y=[11,16,9]
x2=[3,9,11]
y2=[6,15,7]
plt.bar(x,y,color='r')
plt.bar(x2,y2,color='g')
plt.title('Bar graph')
plt.ylabel('Y axis')
plt.xlabel('Xaxis')
plt.show()
```
![image](https://github.com/user-attachments/assets/91072195-29a4-476a-ab21-3c7b75a43267)

```
import matplotlib.pyplot as plt
ages=[2,5,70,40,30,45,50,45,43,40,44,60,7,13,57,18,90,77,32,21,20,40]
range=(0,100)
bins=10
plt.hist(ages,bins,range,color='lightblue',histtype='bar',rwidth=0.8)
plt.xlabel('age')
plt.ylabel('No. of people')
plt.title('My histogram')
plt.show()
```
![image](https://github.com/user-attachments/assets/71baf860-4aab-4688-93b1-68611e6112a2)

```
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
data=np.random.normal(loc=0,scale=1,size=100)
data
```
![image](https://github.com/user-attachments/assets/76c38ec3-e38d-457d-a451-95775634f1ae)

```
fig,ax=plt.subplots()
ax.boxplot(data)
ax.set_xlabel('Data')
ax.set_ylabel('Values')
ax.set_title('Box Plot')
```
![image](https://github.com/user-attachments/assets/29eb7ae8-3aba-4dee-b35d-ae21c97a3b69)

```
labels='Python','C++','Ruby','Java'
sizes=[215,130,245,210]
colors=['gold','yellowgreen','lightcoral','lightskyblue']
explode=(0,0.4,0,0.5)
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True)
plt.axis('equal')
plt.show()
```
![image](https://github.com/user-attachments/assets/60ceb6bf-188f-4ef1-9c8f-9cb3af030d63)

```
activities=['eat','sleep','work','play']
slices=[3,7,8,6]
colors=['r','y','g','b']
plt.pie(slices,labels=activities,colors=colors,startangle=90,shadow=True,explode=(0,0,0.1,0),radius=1.2,autopct='%1.1f%%')
plt.legend()
```
![image](https://github.com/user-attachments/assets/e9fbc20a-d7b2-4a5f-8b85-ba260efa4ef9)


# Result:
Thus, The implementation of data visualization using matplotlib has been successfully verified.
