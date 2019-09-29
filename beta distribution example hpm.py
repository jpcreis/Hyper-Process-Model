
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

n_components_l = [3,4,5,6]
hyper_degree_l = [3,4,5,6]
#n_components_l = [4]
#hyper_degree_l = [4]

# datapoints for each source model
num_dp = 20

# datapoints to sample from trained source models that corresponds to number of points per shape
num_dp_shape = 200

HPM = True

if HPM:
    file_name = 'hyper_parameter_tuning_hpm.txt'
    file = open(file_name, 'a+')
    file.write('n_comp,degree,mean,std,score_hyper\n')
else:
    file_name = 'hyper_parameter_tuning.txt'
    file = open(file_name, 'a+')
    file.write('model_degree,degree,mean,std,score_hyper\n')

file.close()

# Alpha and Beta values for training
a = [0.5, 1, 5, 10, 15]
b = [0.5, 1, 5, 10, 15]

# Alpha and Beta values for test
values_test = [4, 6, 8, 12]


###########################################################################################
###########################################################################################

comb = []

for i in range(len(a)):
    for j in range(len(b)):

        #if (a[i] == 1 and b[j] == 1):
        #    continue
        if len(comb) == 0:
            comb = np.matrix([[a[i], b[j]]])
        else:
            comb = np.append(comb, [[a[i], b[j]]], axis=0)

print(comb)
print('shape: ', comb.shape[0])

##########################################################################
##########################################################################
# In[2]:

x = np.linspace(0.01, 0.99, num_dp)
x_disp = np.linspace(0.01, 0.99, num_dp_shape)

'''
for i in range(comb.shape[0]):
    plt.plot(x, beta.pdf(x, comb[i,0], comb[i,1]), lw=1, alpha=0.6, label='beta pdf')
plt.show()
'''

##########################################################################
##########################################################################
# In[3]:

from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error


def func(x, a, c, d):
    return a * np.exp(-c * x) + d

def gaus(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def coshyp(x, a, c, d):
    return a * np.cosh(-c * x) + d

# Hyper-Process Modeling implementation
if HPM:

    # Get all shapes ready for SSM
    shapes = []

    for i in range(comb.shape[0]):

        res = -1

        print('Target: ', comb[i, 0], ',', comb[i, 1])

        y = beta.pdf(x, comb[i, 0], comb[i, 1])

        if ((comb[i, 0] == 0.5) and (comb[i, 1] == 0.5)):
            degree = 7
            print('Polynomial Function degree', degree)
            # Polynomial fitting
            coef, _, _, _, _ = np.polyfit(x, y, degree, full=True)
            regr = np.poly1d(coef)

        elif (((comb[i, 0] < 1) and (comb[i, 1] >= 1)) or ((comb[i, 1] < 1) and (comb[i, 0] >= 1))):
            print('Exponential Function')
            # Exponential fitting
            popt, pcov = curve_fit(func, x, y, p0=[1, 1e-6, 1], maxfev=2000)

            regr = lambda x: func(x, *popt)

        elif ((comb[i, 0] > 2) or (comb[i, 1] > 2)):
            print('Gaussian Function')
            # linear fitting
            n = len(x)  # the number of data
            mean = sum(x * y) / n  # note this correction
            sigma = sum(y * (x - mean) ** 2) / n

            popt, pcov = curve_fit(gaus, x, y, p0=[1, mean, sigma])
            regr = lambda x: gaus(x, *popt)

        else:
            degree = 7
            print('Polynomial Function degree', degree)
            # Polynomial fitting
            coef, _, _, _, _ = np.polyfit(x, y, degree, full=True)
            regr = np.poly1d(coef)

        # Calculate error
        res = mean_squared_error(regr(x_disp), beta.pdf(x_disp, comb[i, 0], comb[i, 1]))
        print('Error:', res)

        # Sample 100 datapoints from the trained source models
        if len(shapes) == 0:
            shapes = np.matrix(regr(x_disp))
            # shapes = np.matrix(beta.pdf(x_disp, comb[i,0], comb[i,1]))
        else:
            shapes = np.vstack((shapes, regr(x_disp)))
            # shapes = np.vstack((shapes,beta.pdf(x_disp, comb[i,0], comb[i,1])))

        # print(x)
        # print(y)

        '''
        plt.plot(x, y, '--', lw=1, alpha=0.6, label='beta pdf')
        plt.plot(x_disp, regr(x_disp), lw=1, alpha=0.6, label='beta pdf')
        plt.plot(x_disp, beta.pdf(x_disp, comb[i,0], comb[i,1]), lw=1, alpha=0.6, label='beta pdf')
        plt.title("Beta PDF - alpha: {0}, beta: {1} - HPM".format(comb[i,0], comb[i,1]))
        plt.show()
        '''

    print(shapes)
    print('Shapes shape: ', shapes.shape)



##########################################################################
##########################################################################
# In[4]:

from sklearn.metrics import mean_squared_error


#############################################
# SSM

def decomposition(models):

    print('Start Decomposing...')

    covariance = np.cov(models)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    print('Decomposing complete!')

    return eigenvalues.real, eigenvectors.real

def get_suitable_eigen(eigenvals):

    variance_max = 1
    max_eigen = n_components

    sum_eigenvals = sum(eigenvals)
    variance = eigenvals / sum_eigenvals
    variance = [value if np.abs(value) > 0.00001 else 0 for value in variance]

    comulative_sum = 0

    for i in range(0, len(eigenvals)):
        comulative_sum += variance[i]

        print('Comulative sum: ', comulative_sum)
        print('variance: ', variance[i])

        # if (comulative_sum >= variance_max and i > 0):
        if (comulative_sum >= variance_max):
            return i + 1

        if i + 1 == max_eigen:
            return i + 1
    return -1

    return max_eigen

def get_b_param(mean, model, evec):
    sub = (model - mean)
    return np.dot(np.transpose(evec), np.transpose(sub))

def generate_model(mean, evec, b):

    return mean + np.transpose(np.dot(evec, b))


# Hyper parameters optimization
for hyper_degree in hyper_degree_l:
    for n_components in n_components_l:

        from sklearn import datasets, linear_model
        import matplotlib.pyplot as plt
        from sklearn.metrics import mean_squared_error

        if HPM:

            ################################################################
            # Mean model
            mean_model = np.mean(shapes, axis=0)

            print(mean_model)
            print('Mean model shape: ', np.array(mean_model).shape)

            # eigenvectors
            eigenvalues, eigenvectors = decomposition(shapes.transpose())

            print('eigenvectors shape: ', eigenvectors.shape)

            # suitable eigenvectors
            # modes_def = get_suitable_eigen(eigenvalues)
            modes_def = n_components

            print('# suitable eigenvals: ', modes_def)

            # filter suitable eigenvectors
            eigenvalues = eigenvalues[0:modes_def]
            eigenvectors = np.transpose(np.transpose(eigenvectors)[0:modes_def])

            print('eigenvalues shape: ', np.array(eigenvalues).shape)
            print('eigenvectors shape: ', np.array(eigenvectors).shape)

            models_b = []
            recons_error = []

            # Get all the deformable parameters
            for i in range(shapes.shape[0]):
                # Calculate b
                # new_b = self.metaprocess.get_b_param(mean_model, preds_meta[i], eigenvectors)
                new_b = get_b_param(mean_model, shapes[i], eigenvectors)

                print("eigenvectors shape: ", eigenvectors.shape, " new_b: ", new_b.shape)
                gen_shape = generate_model(mean_model, eigenvectors, new_b)

                print("gen_shape: ", gen_shape)
                print("gen_shape shape: ", gen_shape.shape)

                gen_error = mean_squared_error(gen_shape, shapes[i])

                print("Condition: ", comb[i], ", Mean Squared Error: ", gen_error)

                recons_error = np.append(recons_error, gen_error)

                models_b.append(np.array(new_b).flatten())
                # proc_req_y.append(process_req_metamodel[d_name])

                print(new_b, ' Process Req: ', comb[i])

                '''
                plt.plot(np.array(x_disp),np.array(shapes[i])[0], '--' , lw=1, alpha=0.6, label='beta pdf')
                plt.plot(np.array(x_disp),np.array(gen_shape)[0], lw=1, alpha=0.6, label='beta pdf')
                plt.show()
                '''

            print('df_models_b: ', np.matrix(models_b).shape)
            print('Reconstruction error mean: ', recons_error.mean())
            print('Reconstruction error std: ', recons_error.std())

            ##########################################################################
            ##########################################################################
            # In[5]:

            models_b = np.matrix(models_b)

            # print("Conditions: " , comb)
            print("Conditions shape: ", comb.shape)
            # print("Coefficients: " , models_b)
            print("Coefficients shape: ", models_b.shape)

            hyper_model = MultiOutputRegressor(make_pipeline(PolynomialFeatures(hyper_degree),
                                                             linear_model.LinearRegression(fit_intercept=True,
                                                                                           normalize=True)))

            ##################################################################
            # Predict
            hyper_model.fit(comb, models_b)
            pred = hyper_model.predict(comb)

            score = hyper_model.score(comb, models_b)

            print("Score: ", score)

        # Hyper-Model implementation
        else:

            coeffs = []
            error_models = []

            for i in range(comb.shape[0]):

                print('Target: ', comb[i, 0], ',', comb[i, 1])

                y = beta.pdf(x, comb[i, 0], comb[i, 1])

                coef, _, _, _, _ = np.polyfit(x, y, n_components, full=True)
                regr = np.poly1d(coef)

                res = mean_squared_error(regr(x_disp), beta.pdf(x_disp, comb[i, 0], comb[i, 1]))

                error_models = np.append(error_models, res)

                print('Error:', res)
                print('Coefficients:', coef)

                if len(coeffs) == 0:
                    coeffs = np.matrix(coef)
                else:
                    coeffs = np.append(coeffs, [coef], axis=0)

                '''
                plt.plot(x, y, '--', lw=1, alpha=0.6, label='beta pdf')
                plt.plot(x_disp, regr(x_disp), lw=1, alpha=0.6, label='beta pdf')
                plt.plot(x_disp, beta.pdf(x_disp, comb[i,0], comb[i,1]), lw=1, alpha=0.6, label='beta pdf')
                plt.title("Beta PDF - alpha: {0}, beta: {1} - HM".format(comb[i,0], comb[i,1]))
                plt.show()
                '''


            print('All coefficients: ', coeffs)
            print('Mean error models: ', error_models.mean())

            #hyper-model

            hyper_model = MultiOutputRegressor(make_pipeline(PolynomialFeatures(hyper_degree),
                                                             linear_model.LinearRegression(fit_intercept=True,
                                                                                           normalize=True)))

            hyper_model.fit(comb, coeffs)
            pred = hyper_model.predict(comb)

            score = hyper_model.score(comb, coeffs)

            print("Score: ", score)

        ##########################################################################
        ##########################################################################
        # In[7]:

        import matplotlib.pyplot as plt

        # Perform tests to compare the performance of both approaches

        tests = []
        final_results = []

        for i in range(len(values_test)):
            for j in range(len(values_test)):
                if len(tests) == 0:
                    tests = np.matrix([[values_test[i], values_test[j]]])
                else:
                    tests = np.append(tests, [[values_test[i], values_test[j]]], axis=0)

        print("Tests: " , tests)
        print("Tests shape: " , tests.shape)

        print("Number of Components: " , n_components , ", Hyper Degree: " , hyper_degree)

        for i in range(len(tests)):

            if HPM:
                prediction_res = hyper_model.predict(np.matrix(tests[i]))
                result_def = prediction_res[0]

                # print(result_def)

                new_gen_shape = generate_model(mean_model, eigenvectors, result_def)
                new_gen_shape = np.array(new_gen_shape)[0]

                # print('New shape shape: ' , new_gen_shape.shape)
                # print('New shape: ' , new_gen_shape)

                final_error = mean_squared_error(beta.pdf(x_disp, tests[i, 0], tests[i, 1]), new_gen_shape)

                # print('HMP error: ' , final_error)

                print(tests[i], ' -> Error for standard approach: ', final_error)

                final_results = np.append(final_results, final_error)

                '''
                plt.plot(x_disp, beta.pdf(x_disp, tests[i,0], tests[i,1]), '--', lw=1, alpha=0.6, label='beta pdf')
                plt.plot(x_disp, new_gen_shape, lw=1, alpha=0.6, label='beta pdf')
                plt.title("Beta PDF - alpha: {0}, beta: {1} - HPM Test\nMSE={2}".format(tests[i,0], tests[i,1], final_error))
                plt.show()
                '''

            else:

                prediction_res = hyper_model.predict(np.matrix(tests[i]))
                result = prediction_res[0]

                regr_new = np.poly1d(result)
                final_error = mean_squared_error(beta.pdf(x_disp, tests[i, 0], tests[i, 1]), regr_new(x_disp))

                print(tests[i], ' -> Error for standard approach: ', final_error)

                final_results = np.append(final_results, final_error)

                '''
                plt.plot(x_disp, beta.pdf(x_disp, tests[i,0], tests[i,1]), '--', lw=1, alpha=0.6, label='beta pdf')
                plt.plot(x_disp, regr_new(x_disp), lw=1, alpha=0.6, label='beta pdf')
                plt.title("Beta PDF - alpha: {0}, beta: {1} - HP Test\nMSE={2}".format(tests[i, 0], tests[i, 1],final_error))
                plt.show()
                '''

        final_results = np.array(final_results)

        print('Mean: ', final_results.mean())
        print('STD: ', final_results.std())

        ##############################################
        # temp writing
        file = open(file_name, 'a+')
        file.write('{0},{1},{2:.3f},{3:.3f},{4:.3f}\n'.format(n_components, hyper_degree, final_results.mean(),
                                                              final_results.std(), score))
        file.close()
        ##############################################
