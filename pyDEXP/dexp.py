"""
Author: Ray Gunawidjaja
Date: 08/23/2023

Description:
- Generate a design matrix for a 2-level full factorial design.
- Calculate, analyze, and determine the significant effects using Pareto chart and ANOVA.
- Calculate the coded linear regression model, as well as the residuals.  

Limitation: Only for 2-level full factorial designs

Example data is adapted from: 
- Mark J. Anderson and Patrick J. Whitcomb, "DOE Simplified: Practical Tools for Effective Experimentation", Productivity, Inc., 2000.
"""

import itertools
from itertools import combinations
import pandas as pd
import numpy as np

from collections import OrderedDict
import matplotlib.pyplot as plt

import scipy.stats as stats



class FullFactorial:
    def __init__(self, dict_factor_levels, no_of_response=1):

        self.dict_factor_levels = OrderedDict(dict_factor_levels)
        self.no_of_response = no_of_response

        symbolic_factor = itertools.cycle(chr(i) for i in range(ord('A'), ord('Z')+1))
        self.dict_symbolic_factor_levels=OrderedDict()

        self.dict_symbolic_factor_levels = OrderedDict(
            (next(symbolic_factor), [-1, 1]) for _ in dict_factor_levels.keys()
            )

        #check
        #print(self.dict_symbolic_factor_levels)

            
    def generate_exp_df(self, option="actual"):
        """Generate a dataframe of the various run combinations

           input:
            - option (str): choose between "actual" and "symbolic" ([-1, 1]).
            - factor_levels (dict): a dictionary containing a list of two elements corresponding to the low and high levels.
            - no_of_response (int): any values equal or larger than 2 to give meaningful output.

           output:
            - df (dataframe): a dataframe of experimental runs 
        """
        
        if option=="actual":
            factor_names = list(self.dict_factor_levels.keys())
            levels = list(self.dict_factor_levels.values())

        elif option=="symbolic":
            factor_names = list(self.dict_symbolic_factor_levels.keys())
            levels = list(self.dict_symbolic_factor_levels.values())

        # Generate all possible combinations of factor levels
        combinations = list(itertools.product(*levels))

        lst_response = ["R%s" % (i+1) for i in range(self.no_of_response)]
        columns = ['Run'] + factor_names + lst_response

        # Generate rows using list comprehension
        data = [
            [i] + list(combo) + [0] * self.no_of_response
            for i, combo in enumerate(combinations, start=1)
        ]

        # Create a pandas DataFrame
        exp_df = pd.DataFrame(data, columns=columns)

        return exp_df
    

    def generate_interactions_df(self, dict_response):
        """Generate dataframe of interaction columns for calculating effects

           input:
            - dict_response (dict): each key is a list of the measured responses.

           return:
            - df_new (dataframe): a dataframe of -1 and 1 for main and interaction factors.
            - dict_R_effect (dict): a dictionary of n responses. Each key is a dictonary of the main 
                              and interaction factor effects.
            - dict_R_avg_neg_and_pos (dict): a dictionary of n responses. Each key is a dictonary of the main 
                              and interaction factor average negative and positive effects.
        """

        result = []

        #make a symbolic dataframe
        no_of_response=len(dict_response.keys())
        df=self.generate_exp_df("symbolic")

        factors=df.columns[1:-no_of_response] #exclude the run no and the responses columns
        num_factors = len(df.columns)

        df_new = df.iloc[:, :-no_of_response].copy()  # Create a copy to avoid modifying the original DataFrame

        #for two or more interactions: 
        # make a list of factors
        for r in range(2, num_factors + 1):
            comb = combinations(factors, r)
            result.extend([''.join(c) for c in comb])

        # make columns of interaction factors with -1s and 1s in each cell.
        for interaction in result:
            interaction_values = np.prod(df[list(interaction)], axis=1)
            
            df_new.loc[:, interaction] = interaction_values  # Use .loc for assignment

        
        # calculate the effects and save to a dictionary
        dict_R_effects=OrderedDict()
        dict_R_avg_neg_and_pos=OrderedDict()

        #iterate over each response
        for k,v in dict_response.items(): 
            dict_effects=OrderedDict()
            dict_avg_neg_and_pos=OrderedDict()

            for col in df_new.columns[1:]:
                #1) calculate average_negative and average_positive
                avg_neg, avg_pos = self.__calculate_product_and_averages(np.array(df_new[col]), np.array(v))
                #save results
                dict_avg_neg_and_pos[col]=(avg_neg, avg_pos)

                #2) calculate the product between the symbolic and response columns 
                #effect = arr_avg_pos-arr_avg_neg
                #save results
                dict_effects[col]=avg_pos - avg_neg #effect
            
            #add to dict_R
            dict_R_effects[k]=dict_effects
            dict_R_avg_neg_and_pos[k]=dict_avg_neg_and_pos 

        #at the end, add response to the dataframe
        for k,v in dict_response.items():
            df_new[k]=v

        return df_new, dict_R_effects, dict_R_avg_neg_and_pos
    

    def __calculate_product_and_averages(self, arr1, arr2):
        """Calculate the product between symbolic values (-1 and 1)
           and measured response.

           input:
            - arr1 (np-array): -1s and 1s 
            - arr2 (np-array): measured response

           output:
            - average_negative and average_positive (float): average values
        """

        # Initialize lists to collect values for -1 and 1 multipliers
        arr_neg = []
        arr_pos = []
        
        # collect the elements according to the symbols
        for val1, val2 in zip(arr1, arr2):
            if val1 == -1:
                arr_neg.append(val2)
            elif val1 == 1:
                arr_pos.append(val2)
        
        # Calculate the average of values for -1 and 1 multipliers
        avg_neg = np.mean(arr_neg)
        avg_pos = np.mean(arr_pos)
        
        return avg_neg, avg_pos


    @staticmethod
    def save_exp_df(exp_df):
        """Save experimental dataframe to a csv file

           input:
            - exp_df (df): a dataframe of an experimental run

           output:
            - a csv file  
        """

        #data will be saved in the same directory as this python file.
        filename = input("Please enter the filename to save the design matrix: ")
        exp_df.to_csv("%s.csv" %filename, index=False)

        print("Data has been saved to: %s.csv" %(filename))


    @staticmethod
    def randomize_order(exp_df):
        """Given a dataframe of experimental runs, randomize it

           input:
            - exp_df (df): a dataframe of the experimental runs

           output:
            - exp_df_randomized: a randomized dataframe.
        """

        randomized_exp_df=exp_df.sample(frac=1, random_state=42)

        return randomized_exp_df.sample(frac=1, random_state=42)


    @staticmethod
    def perform_anova(dict_effects, lst_significant_effects):
        """Calculate sum of squares, mean squares, and the corresponding F-values

            input:
             - dict_effects (dict): a dictionary of effects, where each key is an (main and multiple factor) 
                                    effect and each value is the corresponding value. 
             - lst_significant_effects (list): a list of effects that are to be tested
                                    for significance. These are previously determined 
                                    from the half-normal plot.
            output:
             - ANOVA analysis
        """

        dict_SS=OrderedDict()
        df_anova=pd.DataFrame()
        
        print(dict_effects)

        #calculate SS and MS for significant effects.
        #dfs are 1 and SS=MS
        N=len(dict_effects.keys())+1

        DF_res=0 #degrees of freedom for model
        SS_res=0

        DF_model=0
        SS_model=0


        for k in dict_effects.keys():
            if k in lst_significant_effects:
                dict_SS[k]=((dict_effects[k])**2)*(N/4)
                DF_model+=1
                SS_model+=dict_SS[k]

            else:
                DF_res+=1
                SS_res+=((dict_effects[k])**2)*(N/4)

        MS_res=SS_res/DF_res
        df_anova["Source"] = ["Model"] + list(dict_SS.keys()) + ["Residual"] + ["Cor Total"]
        df_anova["SS"]=[SS_model]+list(dict_SS.values())+[SS_res]+[SS_model+SS_res]
        df_anova["DF"]=[DF_model]+len(dict_SS.keys())*[1]+[DF_res]+[DF_model+DF_res]
        df_anova["MS"]=(df_anova["SS"]/df_anova["DF"]).round(1)
        df_anova["F-value"]=(df_anova["MS"]/MS_res).round(1)

        #calculate Prob>F for 1% or 0.01 risk
        risk = 0.01  # 1% risk level
        dfn = 3     # Degrees of freedom numerator
        dfd = 20    # Degrees of freedom denominator

        f_value = stats.f.ppf(1 - risk, dfn, dfd)
        df_anova["Crit_F"]=(df_anova["DF"]/DF_res).round(4)
        
        # Compare the values of Crit_F and F-value
        output = ["<0.01" if c1 > c2 else "" for c1, c2 in zip(df_anova['F-value'], df_anova['Crit_F'])]
        df_anova["Prob>F"]=output

        #slice output
        df_anova.loc[df_anova.index[-1:], "MS"] = ""
        df_anova.loc[df_anova.index[-2:], "F-value"] = ""
        df_anova.loc[df_anova.index[-2:], "Crit_F"] = ""
        df_anova.loc[df_anova.index[-2:], "Prob>F"] = ""
        
        #report results
        print("")
        print("ANOVA")
        print("-----------------------------------------")
        print(df_anova)
        

    @staticmethod
    def coded_linear_regression(data):
        """Perform coded linear regression analysis on two-level full factorial data.

        input:
         - data (dict): A dictionary containing 'X1', 'X2', and 'Y' columns.

        return:
         - dict: A dictionary containing regression coefficients for 'intercept', 'X1', and 'X2'.
        """

        independent_vars = list(data.keys())
        independent_vars.remove('Y')  # Remove the dependent variable key

        # Extract data
        Y = np.array(data['Y'])
        X_coded = np.column_stack([np.ones_like(Y)] + [np.array(data[var]) for var in independent_vars])

        # Perform linear regression using the normal equation
        coefficients = np.linalg.inv(X_coded.T @ X_coded) @ X_coded.T @ Y

        results = {'intercept': coefficients[0]}
        for i, var in enumerate(independent_vars):
            results[var] = coefficients[i + 1]

        return results


    @staticmethod
    def calculate_effects(data):
        """Plot effects for one or two factors.

        input:
         - data (dict): A dictionary containing factor names as keys and their corresponding values as lists. 
                        The last key is the response variable 'Y'.
        return:
         - None (displays the plot).
        """

        factors = list(data.keys())
        factors.remove('Y')  # Remove the response variable key

        if len(factors) == 1: #one factor
            factor_values = data[factors[0]]
            avg_neg = np.mean([data['Y'][i] for i, val in enumerate(factor_values) if val == -1])
            avg_pos = np.mean([data['Y'][i] for i, val in enumerate(factor_values) if val == 1])
            
            plt.plot([-1, 1], [avg_neg, avg_pos], marker='o')
            plt.xlabel(factors[0])
            plt.ylabel('Response')
            plt.title('One Factor Effects')
        
        elif len(factors) == 2: #two factors
            factor_A_values = data[factors[0]]
            factor_B_values = data[factors[1]]
            
            avg_A_neg_B_neg = np.mean([data['Y'][i] for i, (a, b) in enumerate(zip(factor_A_values, factor_B_values)) if a == -1 and b == -1])
            avg_A_neg_B_pos = np.mean([data['Y'][i] for i, (a, b) in enumerate(zip(factor_A_values, factor_B_values)) if a == -1 and b == 1])
            avg_A_pos_B_neg = np.mean([data['Y'][i] for i, (a, b) in enumerate(zip(factor_A_values, factor_B_values)) if a == 1 and b == -1])
            avg_A_pos_B_pos = np.mean([data['Y'][i] for i, (a, b) in enumerate(zip(factor_A_values, factor_B_values)) if a == 1 and b == 1])
            
            plt.plot([-1, 1], [avg_A_neg_B_neg, avg_A_neg_B_pos], marker='o', label=f'{factors[0]} = -1')
            plt.plot([-1, 1], [avg_A_pos_B_neg, avg_A_pos_B_pos], marker='o', label=f'{factors[0]} = 1')
            plt.xlabel(factors[1])
            plt.ylabel('Response')
            plt.title('Two Factor Effects')
            plt.legend()
        
        plt.xticks([-1, 1])
        plt.show()


    @staticmethod
    def calculate_residuals_and_predictions(dict_coefficients, dict_data):
        """Calculate residuals and predicted values using a coded linear equation.

        input:
         - dict_coefficients (dict): A dictionary containing coefficient values for the coded linear equation.
         - dict_data (dict): A dictionary containing factor names as keys and their corresponding values as lists.
                        The last key is the response variable 'Y'.

        return:
         - dict: A dictionary containing 'residuals' and 'predictions'.
        """

        residuals = []
        predictions = []

        for i in range(len(dict_data['Y'])):
            prediction = dict_coefficients['intercept']
            for factor, value in dict_data.items():
                if factor != 'Y':
                    prediction += dict_coefficients[factor] * value[i]
            residuals.append(dict_data['Y'][i] - prediction)
            predictions.append(prediction)

        return {'residuals': residuals, 'predictions': predictions}


    @staticmethod
    def generate_pareto_chart(dict_data):
        """Given a dictionary of effects, generate a pareto chart to determine the significant effects.
        The threshold is set to 80%.

        input:
            - dict_data (dict): the keys are the factors and the values are the effects.

        output:
            - Pareto chart (plot).
        """

        factors = list(dict_data.keys())
        effects = list(dict_data.values())
        
        # Take the absolute value of the effects
        abs_effects = [abs(effect) for effect in effects]
        
        # Sort factors and corresponding effects by absolute effect values
        sorted_factors, sorted_abs_effects = zip(*sorted(zip(factors, abs_effects), key=lambda x: x[1], reverse=True))
        
        # Calculate cumulative percent total
        total_effect = sum(sorted_abs_effects)
        cumulative_percent_total = [sum(sorted_abs_effects[:i+1]) / total_effect * 100 for i in range(len(sorted_abs_effects))]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Bar plot for absolute effects
        ax1.bar(sorted_factors, sorted_abs_effects, color='blue')
        ax1.set_xlabel('Factors')
        ax1.set_ylabel('Absolute Effects', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2 = ax1.twinx()
        
        # Line plot for cumulative percent total
        ax2.plot(sorted_factors, cumulative_percent_total, color='orange', marker='o', label='Cumulative % Total')
        ax2.axhline(80, color='red', linestyle='--', linewidth=1, label='80% Threshold')
        ax2.set_ylabel('Cumulative % Total', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        fig.tight_layout()
        
        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        
        plt.title('Pareto Chart with Cumulative Percent Total')
        plt.xticks(rotation=45)
        plt.show()
        
#-----------------------------------------------------------------------------------------------


"""
print("")
print("Example 1: Generate an experimental design")
print("")
factor_levels = {
    'A': ['cheap', 'costly'],
    'B': [4, 6],
    'C': [75, 100],
}

no_of_response=1

full_factorial = FullFactorial(factor_levels, no_of_response)
exp_df = full_factorial.generate_exp_df("actual")

# Print the generated pandas DataFrame
print(exp_df)
print("")

# Randomize the order of experiments
exp_df_randomize = full_factorial.randomize_order(exp_df)
print(exp_df_randomize)
print("")

#save dataframe to a csv
FullFactorial.save_exp_df(exp_df_randomize)

print("")
print("Example 2: Input response and calculate effects")
print("")
df_new, dict_R_effects, dict_R_avg_neg_and_pos = full_factorial.generate_interactions_df({'Taste': [74, 81, 71, 42, 75, 77, 80, 32]})

print(df_new)
print("")
print(dict_R_effects)
print("")
print(dict_R_avg_neg_and_pos)

print("")
print("Example 3: plot Pareto chart")
print("")
for k, v in dict_R_effects.items():
    print(k)
    dict_output=FullFactorial.generate_pareto_chart(v)

    print(dict_output)

print("")
print("Example 4: perform anova analysis")
print("")

#get just one response
myDict=dict_R_effects[list(dict_R_effects.keys())[0]]

full_factorial.perform_anova(myDict, ['C', 'B' ,'BC'])

print("")
print("Example 5: perform coded linear regression analysis")
print("")

data = {
    'B': [-1, -1, 1, 1, -1, -1, 1, 1],
    'C': [-1, -1, -1, -1, 1, 1, 1, 1],
    'BC': [1, 1, -1, -1, -1, -1, 1, 1],
    'Y': [74, 75, 71, 80, 81, 77, 42, 32]
}

regression_results = FullFactorial.coded_linear_regression(data)

print(regression_results)

print("")
print("Example 6: plot one- and two-factor effects")
print("")

data_one_factor={
                'C': [-1, -1, -1, -1, 1, 1, 1, 1],
                'Y': [74, 75, 71, 80, 81, 77, 42, 32]
                }
data_two_factors={
                'C': [-1, -1, -1, -1, 1, 1, 1, 1],
                'B': [-1, -1, 1, 1, -1, -1, 1, 1],
                'Y': [74, 75, 71, 80, 81, 77, 42, 32]
                }

# Plot one factor effects
FullFactorial.calculate_effects(data_one_factor)

# Plot two factor effects
FullFactorial.calculate_effects(data_two_factors)

print("")
print("Example 7: calculate residuals and predicted values")
print("")

# Coded linear equation coefficients
equation_coefficients = {
    'intercept': 66.5,
    'B': -10.25,
    'C': -8.5,
    'BC': -10.75
}

# The corresponding coded factors and response data
data = {
    'B': [-1, -1, 1, 1, -1, -1, 1, 1],
    'C': [-1, -1, -1, -1, 1, 1, 1, 1],
    'BC': [1, 1, -1, -1, -1, -1, 1, 1],
    'Y': [74, 75, 71, 80, 81, 77, 42, 32]
}

# Calculate residuals and predicted values
results = FullFactorial.calculate_residuals_and_predictions(equation_coefficients, data)

# Scatter plot of residuals versus predicted values
plt.scatter(results['predictions'], results['residuals'], marker='s')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Predicted Values')
plt.grid()
plt.show()
"""