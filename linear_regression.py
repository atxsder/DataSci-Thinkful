import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
%matplotlib inline

# Read in pandas dataframe

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

# Strip percentage and convert to float
loansData['Interest.Rate'] = [float(interest[0:-1])/100 for interest in loansData['Interest.Rate']]

# Remove 'months' from loan Length values
loansData['Loan.Length'] = [int(length[0:-7]) for length in loansData['Loan.Length']]

# Isolate and take low FICO score
loansData['FICO.Score'] = [int(val.split('-')[0]) for val in loansData['FICO.Range']]

intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

# Assign dependent the dependent variable (Interest Rate)
y = np.matrix(intrate).transpose()

# The independent variables shaped as columns
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()

# Combines independent variables into single matrix
x = np.column_stack([x1,x2])

# Fit model
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

# Display regression results
f.summary()