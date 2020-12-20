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

class HyperProcessModel:

    def __init__(self):
        self.in_shape = None
        self.shapes = []
        self.conditions = []
        self.b_params = []
        self.degree = -1
        self.optimization = False
        self.hyper_model = False

    # HPM
    def decomposition(self):
        """
        Performs the decomposition of all shapes using eigenvectors
        :return: all eigenvalues, all eigenvectors
        """
        print('Start Decomposing...')
        models = self.shapes.transpose()

        covariance = np.cov(models)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        print('Decomposing complete!')

        return eigenvalues.real, eigenvectors.real
    # gives preference to the max_variance
    def get_suitable_eigen(self, eigenvals, n_components = None, max_variance = 0.95):
        """
        Select the most suitable eigenvectors to use in the SSM. In this particular case,
        when both number of components and maximum variance are specified, preference is
        given to variance. Iterating through all eigenvectors, if it first reaches the
        target variance, it returns the corresponding eigenvectors, if not, if it reaches
        the number of components, it return the corresponding eigenvectors.
        :param eigenvals: array with all eigenvalurs
        :param n_components: int, default=None - number of components to be included
        :param max_variance: float, default=0.95 - variance to be reached
        :return: int - number of suitable components
        """

        sum_eigenvals = sum(eigenvals)
        variance = eigenvals / sum_eigenvals
        variance = [value if np.abs(value) > 0.00001 else 0 for value in variance]

        comulative_sum = 0

        for i in range(0, len(eigenvals)):
            comulative_sum += variance[i]

            # if (comulative_sum >= variance_max and i > 0):
            if (comulative_sum >= max_variance):
                return i + 1

            if n_components is not None:
                if i + 1 == n_components:
                    return i + 1

        return len(eigenvals)

    def get_b_param(self, mean, shape, evec):
        """
        According to SSM, the b parameters are the deformable parameters that allow to reproduce
        back the original shape using the decomposed shapes (eigenvectors) and mean shape
        :param mean: array (nsamples*nfeatures) - array with all values for mean shape.
        :param shape: array (nsamples*nfeatures) - shape used to calculate b parameters from SSM
        :param evec: ndarray - eigenvectors to be used for the b paramters transformation
        :return: array - b paramters / deformable parameters for the corresponding shape
        """
        sub = (shape - mean)
        return np.dot(np.transpose(evec), np.transpose(sub))

    def get_in_shape(self):
        """
        Return the input used to generate shapes for all process models
        :return: ndarray - (nsamples, nfeatures)
        """
        return self.in_shape

    def generate_shape(self, b):
        """
        Based on a deformable parameter (b), generates the corresponding shape
        :param b: array - set of deformable parameters
        :return: array (nsamples*nfeatures) - generated shape that needs to be reshaped as (nsamples, nfeatures)
        """
        return self.mean_shape + np.transpose(np.dot(self.eigenvectors, b))

    def set_pol_degree(self, degree):
        """
        Set the polynomial degree for the hyper model
        :param degree: int
        :return: None
        """
        self.degree = degree

    def full_factorial_design(self, level, min, max):
        """
        Creates a combination of values, bounded to a minimum and maximum, for a "level" number of combinations
        :param level: int - number of combinations to be produced
        :param min: array (nfeatures) - minimum value for all features
        :param max: array (nfeatures) - maximum value for all features
        :return: ndarray - all combinations
        """

        if level < 2:
            print('Level provided is less than 2')
        elif len(np.array([min])) == 1:
            self.in_shape = np.linspace(min, max, level)
        else:
            array_temp = np.array([])
            factor = len(min)

            for i in range(factor):
                array_temp = np.append(array_temp,np.linspace(min[i],max[i],level))

            matrix_temp = array_temp.reshape(factor,level)
            combinations_temp = list(itertools.product(*matrix_temp))
            result = np.matrix(combinations_temp)

            self.in_shape = result

        return self.in_shape

    def add_shape(self, shape):
        """
        Adds a shape to be used in the SSM. Conditions should be added in the same order as shapes
        :param shape: array (nsamples*nfeatures) - shape to be added
        :return: None
        """
        # Sample 100 datapoints from the trained source models - Produce the shapes
        if len(self.shapes) == 0:
            self.shapes = np.matrix(shape)
        else:
            self.shapes = np.vstack((self.shapes, shape))

    def add_condition(self, cond):
        """
        Adds a certain condition to be used by the hyper model. Shapes should be added in the same order as conditions
        :param cond: array - conditions
        :return: None
        """
        # Sample 100 datapoints from the trained source models - Produce the shapes
        if len(self.conditions) == 0:
            self.conditions = np.matrix(cond)
        else:
            self.conditions = np.vstack((self.conditions, cond))

    def get_mean_shape(self):
        """
        Calculates and returns the mean shape based on all previously added shapes.
        :return: array (nsamples*nfeatures) - Mean shape for SSM
        """
        self.mean_shape = np.mean(self.shapes, axis=0)
        return self.mean_shape

    def get_eigen(self, n_components, max_variance):
        """
        Calculates all eigevectors and eigenvalues and returns only the most suitable ones. Meanwhile, all deformable
        parameters are calculated for all available shapes. This is a combination of previously existing functions to
        automate the calculation process.
        :param n_components: int - number of components
        :param max_variance: float - variance to be reaches
        :return: eigenvalues (array), eigenvectors (ndarray)
        """
        # eigenvectors
        self.eigenvalues, self.eigenvectors = self.decomposition()

        # suitable eigenvectors and not all of them
        modes_def = self.get_suitable_eigen(self.eigenvalues, n_components, max_variance)

        # filter suitable eigenvectors
        self.eigenvalues = self.eigenvalues[0:modes_def]
        self.eigenvectors = np.transpose(np.transpose(self.eigenvectors)[0:modes_def])

        ##########################################
        # Calculate B params
        for i in range(len(self.shapes)):
            if len(self.b_params) == 0:
                self.b_params = np.matrix(self.get_b_param(self.mean_shape, self.shapes[i], self.eigenvectors)).transpose()
            else:
                self.b_params = np.vstack((self.b_params,self.get_b_param(self.mean_shape, self.shapes[i], self.eigenvectors).transpose()))
        ##########################################

        return self.eigenvalues, self.eigenvectors

    def train_hyper_model(self, n_components = None, max_variance = 0.95):
        """
        Train the hyper model. The normal and most interesting scenario is when the number of conditions (c) is
        higher than the number of deformable parameters (b), so the hyper model can be trained as such h: c -> b.
        However, it might be case that b is higher than c, so, ideally the model would be trained
        as such h: b -> c. In this case, either 1) the inverse of h needs to be calculated or 2) an optimization
        problem needs to be formulated to estimate b based on a target c. To ease this latter case,
        a MultiOutputRegressor is used for hyper model, meaning that a model per target is trained, so the case
        of b being higher than c is no longer a problem.
        :param n_components: int - number of components
        :param max_variance: float - variance
        :return: float - score (R^2) of the trained model (do not mistake with error)
        """
        self.get_mean_shape()
        self.get_eigen(n_components, max_variance)

        if self.degree == -1:
            print("Please define first the degree of the Hyper Model")
            exit()

        self.hyper_model = MultiOutputRegressor(make_pipeline(PolynomialFeatures(self.degree),
                                                         linear_model.LinearRegression(fit_intercept=True,
                                                                                       normalize=True)))

        print("Dimension of conditions:" , self.conditions.shape[1])
        print("Dimension of b parameters:" , self.b_params.shape[1])
        ##################################################################
        # Predict
        self.hyper_model.fit(self.conditions, self.b_params)
        score = self.hyper_model.score(self.conditions, self.b_params)

        return score

    def predict(self, new_cond):
        """
        Makes a prediction of the deformable parameters to be used in the SSM based on the new conditions provided.
        :param new_cond: array - new conditions
        :return: array - deformable parameters
        """
        return self.hyper_model.predict(new_cond)

    def get_new_shape(self, new_cond):
        """
        Based on the new conditions, returns the generated shape to be used for further training.
        :param new_cond: array - new conditions
        :return: array (nsamples*nfeatures) - new generated shape
        """
        result_def = self.predict(new_cond)[0]

        new_gen_shape = self.generate_shape(result_def)
        new_gen_shape = np.array(new_gen_shape)[0]

        return new_gen_shape

    def set_hyper_model(self, model):
        """
        This function should be used if a new method needs to be used for the hyper model instead of the
        default polynomial.
        :param model: Predictor
        :return: None
        """
        self.hyper_model = model

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
x_disp = hyper.full_factorial_design(num_dp_shape, 0.01, 0.99) #input X for all shapes

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