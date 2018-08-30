# coding: utf-8


# import libraries
import matplotlib.pyplot as plt
from matplotlib import pylab
import math
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import sklearn.linear_model
from functools import reduce



###
# Structure:
# A. functions
# B. other utilities
###


### A. Methods

# generate data
def generate_data(sample_size=500, number_of_features=6, dist_type = 'uniform', mean=456, sd=45):
    """
    number_of_features = int
    sample_size = int
    dist_type = string
    
    optional args for specifying distributions:
        mean
        sd
        
    returns a numpy dataframe 'data'
    """
    
    if dist_type == 'uniform':
        val_range = list(range(math.ceil(2.5*sample_size*number_of_features)))
        val_range = np.array(val_range)
        data = np.random.choice(a=val_range, replace=True, size=(sample_size,number_of_features))
    
    if dist_type == 'increasing':
        val_range = list(range(math.ceil(2.5*sample_size*number_of_features)))
        val_range = np.array(val_range)
        data = np.random.choice(a=val_range, replace=True, size=(sample_size,number_of_features), p=val_range/val_range.sum())
    
    if dist_type == 'normal':
        data = np.random.normal(loc = mean, scale = sd, size=(sample_size,number_of_features)) 

    if dist_type == 'binormal': # pass in two lists of means
        data1 = np.random.normal(loc = mean[0], scale = sd, size=(sample_size,number_of_features)) 
        data2 = np.random.normal(loc = mean[1], scale = sd, size=(sample_size,number_of_features)) 
        data = np.concatenate((data1,data2),axis=0)
        np.random.shuffle(data) # shuffles along axis 0
        
    if dist_type == 'covariance': # uses generate_response and append_response method
        # initial feature
        data =  generate_data(sample_size=sample_size,number_of_features=1,dist_type = 'normal')
        
        for i in range(number_of_features-1):
            if i % 3 == 1: # generate independent variable
                temp1_data = generate_data(sample_size=sample_size,number_of_features=1,dist_type = 'normal', mean=121, sd=31)
                temp2_data = generate_data(sample_size=sample_size,number_of_features=1,dist_type = 'normal')
                data = append_response(data, response=(temp1_data + temp2_data))
            if i % 3 == 2: # increasing 
                data = append_response(data, response=generate_data(sample_size=sample_size,number_of_features=1,dist_type = 'increasing'))
            else:
                data = append_response(data, response=generate_response(data, coef=[((-1)**int(i/2)) * (5*i / data.shape[1]) for i in range(data.shape[1])]))

#         ### still independent
#         mean = np.random.choice(list(range(-1000,1000)), size=number_of_features*2)
#         sd = np.random.choice(list(range(10,70)), size=number_of_features*2)
#         data = generate_data(sample_size=sample_size, number_of_features=number_of_features*2, dist_type = 'normal', mean=mean, sd=sd)
#         for i in [2*x for x in range(number_of_features)]:
#             data[:,i] += data[:,i+1]
        
#         data = data[:, [2*x for x in range(number_of_features)]]
#         ###
    
    
#     if dist_type == 'mixed':
#         data = np.random.normal(loc = 456, scale = 45, size=(sample_size,number_of_features)) 
        
    return data


# load dataset into individual variables
def load_individual_features(data):
    """
    input = numpy array
    returns a dictionary with np arrays for each feature
    keys = 'featurei'
    """
    feature_dict = {}
    number_of_features=data.shape[1]
    sample_size = data.shape[0]
    for i in range(number_of_features):
#         print('feature' + str(i))
        feature_dict['feature' + str(i)] = [data[j][i] for j in range(sample_size)]
    return feature_dict


# generate response
def generate_response(data, feature_dict=None, coef=None, seed=False):
    """
    function for generating a response variable from data input
    data = numpy array of shape (sample_size, number of features)
    feature_dict = dictionary of lists of observations for each variable
    coeff = list of coefficients for a linear model relationship
    default coeffs = ((-1)**i)*5*(1,2,...,number of vars)
    
    returns the response –– leaves data in place
    use 'append_response' method to combine
    """
    if feature_dict == None: feature_dict = load_individual_features(data) # load if necessary
    if seed == True: np.random.seed(1)
    if coef == None: coef = [ ((-1)**i) * (5*i) for i in range(1,data.shape[1]+1)] # some defaults
        
    sample_size , number_of_features = data.shape # extract info
    assert len(coef) == number_of_features
    # generate errors
    epsilon = np.random.normal(loc=0.0, scale=np.average([np.absolute(x[1]) for x in list(feature_dict.items())])/4, size=sample_size)
    response = np.array([ 
        [sum( [ coef[j]*feature_dict['feature' + str(j)][i] for j in range(number_of_features)] ) + epsilon[i]] 
         for i in range(sample_size)
    ])
    
    return response


# append response to features dataset
def append_response(data, response):
    """
    just a quick method to add a response variable to a dataset
    returns appended dataset with response as last columns / feature
    """
    sample_size , number_of_features = data.shape
    assert sample_size == len(response) # check there is the same number of observations
    appended_data = np.concatenate((data,response), axis=1) # add response column to dataset
    return appended_data


# understand regression along one dimension

def project_data(data, feature_dict=None, retained_feature_index = 0):
    """
    for given dataset, returns a projected dataset onto a single variable, with the other variables padded by mean value
    the purpose is for understanding the regression model on the single retained feature.
    indexing starts at 0
    """
    
    if feature_dict == None: 
        feature_dict=load_individual_features(data)
    
    sample_size , number_of_features = data.shape
    projected_data = np.zeros(shape=(sample_size,number_of_features)) #initialize
    
    means = {}
    for i in range(number_of_features):
        means['feature' + str(i) + 'mean'] = np.mean(feature_dict['feature' + str(i)])
            
    for i in range(sample_size):
        projected_data[i] = np.array(
            [means['feature' + str(j) + 'mean'] for j in range(retained_feature_index)] 
            + [data[i][retained_feature_index]]
            + [means['feature' + str(j) + 'mean'] for j in range(retained_feature_index+1,number_of_features)])
        
    return projected_data



# generate predictions on projected data
def generate_predictions(test_data, trained_model):
    """
    trained_model = (regression) model that has been fit to training data
    test data = for generating predictions
    returns a dictionary of the various projected data 
    and a dictionary of the various predictions for each projected dataset
    keys are:
        'data_projected_onto' + str(i)
        'predictions_along_projection' + str(i)
    
    TODO - modify so you can input a dictionary of already projected 'test datasets'
    """
    
    projected_data_dict = {}
    predictions_dict = {}
    
    sample_size , number_of_features = test_data.shape
    
    for i in range(number_of_features):
        projected_data_dict['data_projected_onto' + str(i)] = project_data(test_data, retained_feature_index=i)
        predictions_dict['predictions_along_projection' + str(i)] = trained_model.predict(projected_data_dict['data_projected_onto' + str(i)])
    
    return (projected_data_dict, predictions_dict)
    


# simple automatted scatterplot of features against response
def response_scatterplot_matrix(data, feature_dict, response=None, desired_indices=None, dictionary_tuple=None):
    """
    simple automatted scatterplot of x_i and y
    
    uses load_individual_features method to get feature_dict
    could put that as an input if you already have it
    
    data = a numpy array of features with shape [sample_size , number_of_features]
    response = measured response var. right now has a default... should consider getting rid of that
    
    uses utilities 
        'prime_factors' 
        reduce from functools
        
    if dictionary_tuple is passed, it pulls projected_data_dict and predictions_dict
    from it and generates a scatterplot matrix with regression prediction lines plotted
    to obtain dictionary_tuple input, use the generate_predictions function.
    keys for projected_data_dict and predictions_dict are:
        'data_projected_onto' + str(i)
        'predictions_along_projection' + str(i)
    """
    
    # initialize necessary variables
    if response.any() == None: response=generate_response(data)
    if feature_dict == None: feature_dict=load_individual_features(data)
    if dictionary_tuple != None:
        _ , predictions_dict = dictionary_tuple
    sample_size , number_of_features = data.shape
    if desired_indices == None:
        desired_indices = list(range(sample_size)) #don't change list indices
    
    # get indices of interest
    plotted_indices = list(set(range(number_of_features)).intersection(list(desired_indices)))
    plotted_indices.sort #in place sorting ... no difference?

    # get plot dimensions
    pfs = prime_factors(len(plotted_indices))
    # cut the list of factors in half, with odds -> lower half gets one more term
    # since lower factors are smaller
    cutoff = int((len(pfs) + 1)/2) 
    if len(pfs) == 1: 
        pfs = [1] + pfs # if prime add in a dimension '1'#
        nrows = int( reduce(lambda x, y: x*y, pfs[0:cutoff]) )
        ncols = int( reduce(lambda x, y: x*y, pfs[cutoff:len(pfs)]) )
    else: 
        nrows = int( reduce(lambda x, y: x*y, pfs[0:cutoff]) )
        ncols = int( reduce(lambda x, y: x*y, pfs[cutoff:len(pfs)]) )

    # create plot array
    ###
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, sharey=True)#, sharex=True)
    plt.subplots_adjust(top=.9, bottom=.1, hspace=.5,wspace=.2) # tentative defaults, tested for n=12 plots
    ax=ax.flatten()    
    ###

    # fill in plot array
    
    # if you have data to plot a regression
    if dictionary_tuple != None: 
        for i in range(len(plotted_indices)):
            ax[i].scatter(feature_dict['feature'+str(plotted_indices[i])], response, s=.5)
            # ax[i].set_title('???' + str(i))
            ax[i].set_xlabel('x'+str(plotted_indices[i]))
            ax[i].set_ylabel('y') 
            ax[i].plot(feature_dict['feature'+str(plotted_indices[i])], predictions_dict['predictions_along_projection' + str(plotted_indices[i])], color='red', linewidth=.8)
            
    # generate a scatterplot
    else:
        for i in range(len(plotted_indices)): # needs to be sorted!
            ax[i].scatter(feature_dict['feature'+str(plotted_indices[i])], response, s=.5)
            # ax[i].set_title('???' + str(i))
            ax[i].set_xlabel('x'+str(plotted_indices[i]))
            ax[i].set_ylabel('y') 


    plt.show()
    


# generate scatterplot matrix of covariates / features
def features_scatterplot_matrix(data, features=None, diagonal='hist'):
    """
    method for generating simple scatterplot matrix for desired features
    data is a numpy array with shape (sample_size,number_of_features)
    defualt has default featuers = [0,1,...,number_of_features-1]
    diagonal argument is used for the scatter_matrix method from pandas.plotting:
        'kde' for kernel density estimation
        'hist' for histogram
    """
    
    if features==None: features = range(data.shape[1])
    
    df = pd.DataFrame(data[:, list(features)], columns = ['x' + str(i) for i in features])
    dim = min(int(1.5*len(features)), 18)
    scatter_matrix(df, alpha = 0.5, figsize = (dim,dim), diagonal = diagonal)
    plt.show()         
    


# demo
def demo():
    
    # generate multivariable data
    sample_size , number_of_features = (500, 6)
    # params can be specified
    # mean = [-71,44,189,-2,61,0]
    # sd = [40,10,17,20,40,5]
    train_data = generate_data(sample_size , number_of_features, dist_type='normal')
    train_feature_dict = load_individual_features(train_data)
    coef = (-20,125,42,11,-97, 203)
    train_response = generate_response(train_data, train_feature_dict, coef=coef)
    print('input:', train_data[0:1], '\n'+'response:', train_response[0:1], end='\n'*3)

    # train a linear model
    lm = sklearn.linear_model.LinearRegression()
    _ = lm.fit(train_data, train_response)

    test_data = generate_data(dist_type='normal') + .06*generate_data(dist_type='increasing')
    test_feature_dict = load_individual_features(test_data)
    test_response = generate_response(test_data, test_feature_dict, coef=coef)
    predictions = lm.predict(test_data)

    print('features: ', [test_feature_dict['feature'+str(i)][0] for i in range(number_of_features)],
      '\n'+'response: ', test_response[0], 
      '\n'+'prediction: ', predictions[0]) 

    #plots
    print('Scatterplot Matrix of Covariates')
    features_scatterplot_matrix(test_data,diagonal='kde')
    print('Selected Feature/Response Scatterplot Matrix')
    response_scatterplot_matrix(test_data,response=test_response,feature_dict=test_feature_dict,
                            # select features
                            #desired_indices=(1,2),
                            # plot regression
                            dictionary_tuple=generate_predictions(test_data, lm))



###


### B. UTILITIES

def prime_factors(n):
    """Returns all the prime factors of a positive integer"""
    factors = []
    d = 2
    while n > 1:
        while n % d == 0:
            factors.append(d)
            n /= d
        d = d + 1
        if d*d > n:
            if n > 1: factors.append(n)
            break
    return factors




