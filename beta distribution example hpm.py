from sklearn import linear_model
import numpy as np
from scipy.stats import beta
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from hyperprocessmodel import HyperProcessModel

def func(x, a, c, d):
    return a * np.exp(-c * x) + d

def gaus(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def coshyp(x, a, c, d):
    return a * np.cosh(-c * x) + d

hyper = HyperProcessModel()
hyper.set_pol_degree(6) # alternatively, a different model for the hyper model can be used by calling the "set_hyper_model" function

num_dp = 50         # datapoints for each source model to create the benchmark dataset
num_dp_shape = 500  # datapoints to sample from trained source models that corresponds to number of points per shape
granularity = 100   # level of detail (granularity) or steps in each feature. Linear space between min and max with n "granularity" values.

##########################################################################################
# Preparing the benchmark dataset. This is not part of the HPM or HM algorithm
# Alpha and Beta values from Beta distribution for training process models
a = [0.5, 1, 5, 10, 11, 15]
b = [0.5, 1, 5, 10, 11, 15]

# Alpha and Beta values from Beta distribution for testing
values_test = [4, 6, 8, 12]

###########################################################################################
# combination of beta and alpha values to generate mulitple datasets
comb = []

for i in range(len(a)):
    for j in range(len(b)):
        comb.append([a[i], b[j]])

print("all combinations:" , comb)
print('combinations shape: ', len(comb))

'''
for i in range(comb.shape[0]):
    plt.plot(x, beta.pdf(x, comb[i,0], comb[i,1]), lw=1, alpha=0.6, label='beta pdf')
plt.show()
'''


x = np.linspace(0.01, 0.99, num_dp) # input values for the benchmark dataset. Beta distribution only has one input variable rangin from 0 to 1
x_disp = hyper.stochastic_factorial_design(granularity, num_dp_shape, [0.01], [0.99]) #input X for all shapes

print(x_disp)

###################################################################################
# Preparing the benchmark dataset for the HPM and training all process models
for c in comb:

    #print('Target: ', c[0], ',', c[1])

    y = beta.pdf(x, c[0], c[1])

    # these set of conditions allow to train a specific method for the type of curves produced by the beta distribution
    # Using different types of methods for hyper-modeling is not possible in "regular" HM
    if ((c[0] == 0.5) and (c[1] == 0.5)):
        degree = 7
        #print('Polynomial Function degree', degree)
        # Polynomial fitting
        coef, _, _, _, _ = np.polyfit(x, y, degree, full=True)
        regr = np.poly1d(coef)

    elif (((c[0] < 1) and (c[1] >= 1)) or ((c[1] < 1) and (c[0] >= 1))):
        #print('Exponential Function')
        # Exponential fitting
        popt, pcov = curve_fit(func, x, y, p0=[1, 1e-6, 1], maxfev=2000)

        regr = lambda x: func(x, *popt)

    elif ((c[0] > 2) or (c[1] > 2)):
        #print('Gaussian Function')
        # linear fitting
        n = len(x)  # the number of data
        mean = sum(x * y) / n  # note this correction
        sigma = sum(y * (x - mean) ** 2) / n

        popt, pcov = curve_fit(gaus, x, y, p0=[1, mean, sigma])
        regr = lambda x: gaus(x, *popt)

    else:
        degree = 7
        #print('Polynomial Function degree', degree)
        # Polynomial fitting
        coef, _, _, _, _ = np.polyfit(x, y, degree, full=True)
        regr = np.poly1d(coef)

    # Calculate error for the trained models
    #res = mean_squared_error(regr(x), beta.pdf(x, comb[i, 0], comb[i, 1]))
    res = mean_squared_error(regr(x), beta.pdf(x, c[0], c[1]))
    #print('Error:', res)

    hyper.add_shape(regr(x_disp))
    hyper.add_condition(c)

    '''
    plt.plot(x, regr(x), lw=1, alpha=0.6, label='beta pdf')
    plt.plot(x, beta.pdf(x, c[0], c[1]), lw=1, alpha=0.6, label='beta pdf')
    plt.title("Beta PDF - alpha: {0}, beta: {1} - HPM".format(c[0], c[1]))
    plt.show()
    '''

###################################################################################
# Decompose shapes for SSM, calculates mean shape and eigenvectors and trains the hyper-model
hyper.train_hyper_model(n_components=4)

tests = []
final_results = []

###########################################################################
# Preparing all combinations for test cases
for i in range(len(values_test)):
    for j in range(len(values_test)):
        tests.append([values_test[i], values_test[j]])

###########################################################################

for t in tests:
    new_gen_shape = hyper.get_new_shape(np.matrix(t))

    final_error = mean_squared_error(beta.pdf(x_disp, t[0], t[1]), new_gen_shape)

    # print('HMP error: ' , final_error)
    print(t, ' -> Error for standard approach: ', final_error)

    final_results = np.append(final_results, final_error)

    plt.plot(x_disp, beta.pdf(x_disp, t[0], t[1]), '--', lw=1, alpha=0.6, label='beta pdf')
    plt.plot(x_disp, new_gen_shape, lw=1, alpha=0.6, label='beta pdf')
    plt.title("Beta PDF - alpha: {0}, beta: {1} - HPM Test\nMSE={2}".format(t[0], t[1], final_error))
    plt.show()