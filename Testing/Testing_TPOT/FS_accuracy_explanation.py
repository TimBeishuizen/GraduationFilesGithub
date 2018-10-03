import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

exp = range(1000)

outcome95 = []
outcome99 = []
outcome975 = []
outcome995 = []
outcome999 = []

for i in exp:
    outcome95.append(0.95 ** i)
    outcome975.append(0.975 ** i)
    outcome99.append(0.99 ** i)
    outcome995.append(0.995 ** i)
    outcome999.append(0.999 ** i)

plt.plot(exp, outcome95, exp, outcome975, exp, outcome99, exp, outcome995, exp, outcome999)
plt.title(r"The correction factor when using a certain number of features following the equation: $\beta$ ^ #features")
plt.xlabel("The number of features")
plt.ylabel("The correction factor")
plt.legend([r'$\beta = 0.95$', r'$\beta = 0.975$', r'$\beta = 0.99$', r'$\beta = 0.995$', r'$\beta = 0.999$'])
plt.xscale("log")
plt.show()