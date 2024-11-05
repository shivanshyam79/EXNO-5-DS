# EXNO-5-DS-DATA VISUALIZATION USING MATPLOT LIBRARY
## NAME : SHYAM R
## REG NO: 212223040200
## Aim:
  To Perform Data Visualization using matplot python library for the given datas.

## EXPLANATION:
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

## ALGORITHM:
STEP 1:Include the necessary Library.

STEP 2:Read the given Data.

STEP 3:Apply data visualization techniques to identify the patterns of the data.

STEP 4:Apply the various data visualization tools wherever necessary.

STEP 5:Include Necessary parameters in each functions.

## CODING AND OUTPUT:

### PLOTTING THE LINE GRAPHS:
```python
import matplotlib.pyplot as plt

x1=[1,2,3]
y1=[2,4,1]

plt.plot(x1,y1,label="line 1")

x2=[1,2,3]
y2=[4,1,3]

plt.plot(x2,y2,label="line 2")

plt.xlabel('x - axis')
plt.ylabel('y - axis')

plt.title('Two lines on same graph!')

plt.legend() 
plt.show()
```

### OUTPUT:
![alt text](screenshot/image.png)

### PLOTTING THE LINE GRAPHS WITH CUSTOMIZATONS:
```PYTHON
import matplotlib.pyplot as plt

x=[1,2,3,4,5,6]
y=[2,4,1,5,2,6]

plt.plot(x,y,color='yellow', linestyle='dashed', linewidth=3, marker='o', markerfacecolor='brown', markersize=12)
plt.ylim(1,8)
plt.xlim(1,8)
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title('Some cool customizations')
plt.show()
```
### OUTPUT:
![alt text](screenshot/image-1.png)

## PLOTTING THE LINE GRAPHS:
```python
import matplotlib.pyplot as plt

years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011]
apples = [0.895, 0.91, 0.919, 0.926, 0.928, 0.931, 0.934, 0.936, 0.937, 0.9375, 0.9372, 0.939]
oranges = [0.962, 0.941, 0.938, 0.923, 0.918, 0.903, 0.897, 0.884, 0.871, 0.868, 0.85, 0.836]

plt.plot(years, apples, marker='o', linestyle='-', color='b')  
plt.plot(years, oranges, marker='s', linestyle='--', color='orange') 

plt.xlabel('Year')
plt.ylabel('Yield (tons per hectare)')

plt.title("Crop Yields in Kanto")
plt.legend(['Apples', 'Oranges'])
plt.show()
```

### OUTPUT:
![alt text](screenshot/image-8.png)

### PLOTTING USING LINE GRAPH:
```PYTHON
import matplotlib.pyplot as plt

yield_apples=[0.854,0.95,0.935,0.945,0.959,0.961]
plt.plot(yield_apples)
```
```PYTHON
years=[2010,2011,2012,2013,2014,2015]
yield_apples=[0.854,0.95,0.935,0.945,0.959,0.961]

plt.plot(years,yield_apples)
```
### OUTPUT:
![alt text](screenshot/image-4.png)

![alt text](screenshot/image-5.png)


### PLOTTING THE SCATTER PLOTS:
```PYTHON
import matplotlib.pyplot as plt

x_values=[0,1,2,3,4,5]
y_values=[0,1,4,9,16,25]
plt.scatter(x_values, y_values, s=30, color="green")
```
```PYTHON
import matplotlib.pyplot as plt

x=[1,2,3,4,5,6,7,8,9,10]
y=[2,4,5,7,6,8,9,11,12,12]

plt.scatter(x,y,label="stars", color="purple", marker="*",s=30)

plt.xlabel('X - axis')
plt.ylabel('Y - axis')

plt.title('My scatter plt!')

plt.legend()

plt.show()
```
### OUTPUT:

![alt text](screenshot/image-6.png)

![alt text](screenshot/image-7.png)

### PLOTTING THE SCATTER PLOTS:
```PYTHON
import numpy as np
import pandas as pd
import numpy as np

x=np.arange(0,10)
y=np.arange(11,21)

x
```
### OUTPUT:

![alt text](screenshot/image-2.png)

```python
y
```
### OUTPUT:
![alt text](screenshot/image-3.png)

```
y=x*x
y
```

### OUTPUT:
![alt text](screenshot/image-9.png)

```PYTHON
plt.plot(x,y,'r*',linestyle='dashed',linewidth=2, markersize=12)
plt.xlabel('X axis')
plt.ylabel('Y axis')

plt.title('2D Diagram')
plt.show()
```

### OUTPUT:
![alt text](screenshot/image-10.png)

```PYTHON
plt.subplot(2,2,1)
plt.plot(x,y,'g--')
plt.subplot(2,2,2)
plt.plot(x,y,'b*--')
plt.subplot(2,2,3)
plt.plot(x,y,'yo')
plt.subplot(2,2,4)
plt.plot(x,y,'ro')
```

### OUTPUT:
![alt text](screenshot/image-11.png)

```PYTHON
np.pi
```

### OUTPUT:
![alt text](screenshot/image-12.png)


### COMPLETE THE X AND Y COORDINATES FOR POINTS ON A SINE CURVE:
```PYTHON
x=np.arange(0,4*np.pi,0.1)
y=np.sin(x)
plt.title("Sine Wave form")

plt.plot(x,y)
plt.show()
```

### OUTPUT:
![alt text](screenshot/image-13.png)

### AREA CHART:
```PYTHON
import matplotlib.pyplot as plt
import numpy as np

x=[1,2,3,4,5]
y1=[10,12,14,16,18]
y2=[5,7,9,11,13]
y3=[2,4,6,8,10]

plt.fill_between(x,y1,color='pink')
plt.fill_between(x,y2,color='brown')

plt.plot(x,y1,color='red')
plt.plot(x,y2,color='blue')

plt.legend(['y1','y2'])

plt.show()
```

### OUTPUT:
![alt text](screenshot/image-14.png)


```PYTHON
plt.stackplot(x,y1,y2,y3,labels=['Line 1','Line 2','Line 3'])
plt.legend(loc='upper left')
plt.title("Stacked Line Chart")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
```

### OUTPUT:
![alt text](screenshot/image-15.png)

### SPLINE CHART:
```PYTHON
from scipy.interpolate import make_interp_spline

x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([2,4,5,7,8,8,9,10,11,12])

spl=make_interp_spline(x,y)

x_smooth=np.linspace(x.min(),x.max(),100)

y_smooth=spl(x_smooth)

plt.plot(x,y,'o',label='data')

plt.plot(x_smooth,y_smooth,'-',label='spline')

plt.legend()
plt.show()
```

### OUTPUT:
![alt text](screenshot/image-16.png)

### BAR GRAPH:
```PYTHON
import matplotlib.pyplot as plt
height=[10,24,36,40,5]
name=['one','two','three','four','five']

c1=['yellow','purple']

plt.bar(name,height,width=0.8,color=c1)

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title("My bar chart")
plt.show()
```

### OUTPUT:
![alt text](screenshot/image-17.png)

```PYTHON
x=[2,9,11]
y=[11,20,3]
x2=[2,8,10]
y2=[5,14,16]

plt.bar(x,y,color='b')
plt.bar(x2,y2,color='r')

plt.title("Bar Graph")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

### OUTPUT:
![alt text](screenshot/image-18.png)

### HISTOGRAM GRAPH:
```PYTHON
import matplotlib.pyplot as plt
ages=[2,5,70,40,30,45,50,45,43,40,44,60,7,13,57,18,90,77,32,21,20,40]
range=(0,100)
bins=10

plt.hist(ages,bins,range,color='violet',histtype='bar',rwidth=0.9)

plt.xlabel('age')
plt.ylabel('No. of People')

plt.title('My Histogram')

plt.show()
```

### OUTPUT:
![alt text](screenshot/image-19.png)

```PYTHON
ages=[2,5,70,40,30,45,50,45,43,40,44,60,7,13,57,18,90,77,32,21,20,40]

plt.hist(ages,bins=10,color='gold',alpha=0.5)

plt.show()
```

### OUTPUT:
![alt text](screenshot/image-20.png)

### BOX PLOT:
```python
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
data=np.random.normal(loc=0,scale=1,size=100)
data
```

### OUTPUT:
![alt text](screenshot/image-21.png)

```PYTHON
fig,ax=plt.subplots()
ax.boxplot(data)
ax.set_xlabel('Data')
ax.set_ylabel('Values')
ax.set_title("Box Plot")
```

### OUTPUT:
![alt text](screenshot/image-22.png)

### PIE CHART:
```python
import matplotlib.pyplot as plt

activities=['eat','sleep','work','play']

slices=[3,7,8,6]

colors=['r','y','g','b']

plt.pie(slices,labels=activities,colors=colors,startangle=90,shadow=True,explode=(0,0,0.1,0), radius=1.2,autopct='%1.1f%%')

plt.legend()
plt.show()
```

### OUTPUT:
![alt text](screenshot/image-23.png)

```python
import matplotlib.pyplot as plt

labels = ['Python', 'C++', 'Ruby', 'Java', 'C']
sizes = [300, 130, 210, 400, 120]  
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'violet']
explode = (0, 0.4, 0, 0.5, 0)  

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)

plt.axis('equal')
plt.show()
```

### OUTPUT:
![alt text](screenshot/image-24.png)


## RESULT:
Thus the Python Program by using Matplot Library is successfully executed.
