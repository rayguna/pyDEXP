### pyDEXP package for 2^k full-factorial design of experiments

#### 1. About the Package

The dexp v1.0.0 module within the pyDEXP package is used for generating and analyzing a 2^k full-factorial design matrix.

In this initial version, the pyDEXP package contains a dexp v1.0.0 module that contains the FullFactorial class. 

The current instance methods are as follows.

- generate_exp_df
- generate_interactions_df
- __calculate_product_and_averages

The staticmethods are as follows.

- save_exp_df
- randomize_order
- perform_anova
- coded_linear_regression
- calculate_effects
- calculate_residuals_and_predictions
- generate_pareto_chart

#### 2. Example Usage

Example 1: Generate experimental design

```
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
full_factorial.save_exp_df(exp_df_randomize)
```

Example 2: Input response and calculate effects

```
df_new, dict_R_effects, dict_R_avg_neg_and_pos = full_factorial.generate_interactions_df({'Taste': [74, 81, 71, 42, 75, 77, 80, 32]})

print(df_new)
print("")
print(dict_R_effects)
print("")
print(dict_R_avg_neg_and_pos)

print("")
```

Example 3: plot Pareto chart

```
for k, v in dict_R_effects.items():
    print(k)
    dict_output=FullFactorial.generate_pareto_chart(v)

    print(dict_output)

print("")
```

Example 4: perform anova analysis

```
#get just one response
myDict=dict_R_effects[list(dict_R_effects.keys())[0]]

full_factorial.perform_anova(myDict, ['C', 'B' ,'BC'])

print("")
```

Example 5: perform coded linear regression analysis

```
data = {
    'B': [-1, -1, 1, 1, -1, -1, 1, 1],
    'C': [-1, -1, -1, -1, 1, 1, 1, 1],
    'BC': [1, 1, -1, -1, -1, -1, 1, 1],
    'Y': [74, 75, 71, 80, 81, 77, 42, 32]
}

regression_results = FullFactorial.coded_linear_regression(data)

print(regression_results)

print("")
```

Example 6: plot one- and two-factor effects

```
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
```

Example 7: calculate residuals and predicted values

```
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
```

~
