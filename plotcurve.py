import numpy as np
import matplotlib.pyplot as plt
# -1/x bce gradient
# 2*(1-x)*np.log(x)- np.power(1-x,2)/x focal gradient
# -1*np.power(x,-2)
x = np.linspace(0.3, 1, 100)
y = -1/x
plt.plot(x,y,'r')

y = -1*np.power(x,-2)
plt.plot(x,y, 'C1')
plt.show()
