import random
from sklearn import linear_model
import numpy as np
import itertools
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures

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

    def stochastic_factorial_design(self, granularity, n_samples, min, max):
        """
        Creates a combination of values, bounded to a minimum and maximum, for a "level" number of combinations
        :param granularity: int - level of detail (granularity) or steps in each feature. Linear space between min and max with n "granularity" values.
        :param n_samples: int - number of samples to be produced
        :param min: array (nfeatures) - minimum value for all features
        :param max: array (nfeatures) - maximum value for all features
        :return: ndarray - all combinations
        """

        if granularity < 2:
            print('Granularity provided is less than 2')
        elif len(np.array(min)) == 1:
            self.in_shape = np.linspace(min, max, granularity).transpose()[0]
        else:
            array_temp = np.array([])
            factor = len(min)

            for i in range(factor):
                array_temp = np.append(array_temp, np.linspace(min[i], max[i], granularity))

            matrix_temp = array_temp.reshape(factor, granularity)

            combo = matrix_temp
            my_sample = []

            while len(my_sample) < n_samples:
                # Choose one random item from each list; that forms an element
                elem = [comp[random.randint(0, len(comp) - 1)] for comp in combo]
                # Using a set elminates duplicates easily
                my_sample.append(elem)

            result = np.matrix(my_sample)

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