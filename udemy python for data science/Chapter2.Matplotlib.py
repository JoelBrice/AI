"""
Visualization with Matplotlib

"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

Year = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]

Temp_I = [0.72,0.61,0.65,0.68,0.75,0.90,1.02,0.93,0.85,0.99,1.02]

plt.plot(Year,Temp_I)
plt.xlabel('Year')
plt.ylabel('Temp_Index')
plt.title('Global Warming',{'fontsize':12,'horizontalalignment':'center'})
plt.show()


Month = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] 

Customer1 = [12,13,9,8,7,8,8,7,6,5,8,10]

Customer2 = [14,16,11,7,6,6,7,6,5,8,9,12]


plt.plot(Month,Customer1)
plt.plot(Month,Customer2)
plt.show()


plt.plot(Month,Customer1,color = 'red', label = 'Customer1', marker = 'o')
plt.plot(Month,Customer2,color = 'blue', label = 'Customer2', marker = '^')
plt.xlabel('Month')
plt.ylabel('Electricity Consumption')
plt.title('Building Consumption')
plt.legend(loc='upper right')
plt.show()

plt.subplot(1,2,1)
plt.plot(Month,Customer1,color = 'red', label = 'Customer1', marker = 'o')
plt.xlabel('Month')
plt.ylabel('Electricity Consumption')
plt.title('Building Consumption of Customer1')
plt.show()

plt.subplot(1,2,2)
plt.plot(Month,Customer2,color = 'blue', label = 'Customer2', marker = '^')
plt.xlabel('Month')
plt.title('Building Consumption of Customer2')
plt.show()


plt.scatter(Month,Customer1,color = 'red',label = 'cutomer1')
plt.scatter(Month,Customer2,color = 'blue',label = 'cutomer2')
plt.xlabel('Month')
plt.ylabel('Electricity Consumption')
plt.title('Scatter Plot of Building Consumption')
plt.grid()
plt.legend()
plt.show()



plt.hist(Customer1, bins = 10, color = 'green')
plt.ylabel('Elec Consumption')
plt.title('Histogram')
plt.show()

plt.bar(Month,Customer1, width = 0.8, color='b')
plt.show()


plt.bar(Month,Customer1,color = 'red',label = 'cutomer1')
plt.bar(Month,Customer2,color = 'blue',label = 'cutomer2')
plt.xlabel('Month')
plt.ylabel('Electricity Consumption')
plt.title('Bar Chart')
plt.legend()
plt.show()

bar_width = 0.4

Month_b = np.arange(12)

plt.bar(Month_b,Customer1, bar_width, color = 'blue', label = 'Customer1')

plt.bar(Month_b+bar_width, Customer2, bar_width, color= 'red', label = 'Customer2')

plt.xlabel('Month')
plt.ylabel('Electricity Consumption')
plt.title('Bar Chart')
plt.xticks(Month_b + (bar_width)/12, ('Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))

plt.legend()
plt.show()


plt.xticks(Month_b + (bar_width)/12, ('Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))

plt.boxplot(Customer1, notch = False, vert = True)

plt.boxplot([Customer1,Customer2], patch_artist=True,
            boxprops = dict(facecolor = 'red',color = 'red'),
            whiskerprops= dict(color = 'green'),
            capprops= dict(color = 'blue'),
            medianprops = dict(color = 'yellow'))

plt.show()







