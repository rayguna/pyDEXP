import sys
import unittest
sys.path.append('..')  # Add the parent directory to the sys.path
from dexp import FullFactorial

from pandas.testing import assert_frame_equal

import pandas as pd

import numpy as np

from collections import OrderedDict

class TestFullFactorial(unittest.TestCase):
    
    def setUp(self):
        dict_factor_levels={'A:Brand': ['Cheap', 'Costly'], 
                                                     'B:Time': [4,6],
                                                     'C:Power':[75,100]}
        self.full_factorial = FullFactorial(dict_factor_levels, 1)


    def test_generate_exp_df(self):
        exp_df = self.full_factorial.generate_exp_df()
        expected_exp_df=pd.DataFrame({'Run': [i+1 for i in range(2**3)],
                                      'A:Brand': ['Cheap', 'Cheap', 'Cheap', 'Cheap', 
                                                  'Costly', 'Costly', 'Costly', 'Costly'], 
                                      'B:Time': [4, 4, 6, 6, 4, 4, 6, 6],
                                      'C:Power':[75, 100, 75, 100, 75, 100, 75, 100],
                                      'R1': (2**3)*[0]})

        assert_frame_equal(exp_df, expected_exp_df)


    def test_calculate_product_and_averages(self):
        exp_df = self.full_factorial.generate_exp_df("symbolic")
        #populate R1 column
        exp_df['R1']= [74, 81, 71, 42, 75, 77, 80, 32]
        avg_neg, avg_pos = self.full_factorial._FullFactorial__calculate_product_and_averages(np.array(exp_df['A']), np.array(exp_df['R1']))

        half_rows=int(len(exp_df)/2)
        

        expected_avg_neg=exp_df['R1'][:half_rows].mean()
        expected_avg_pos=exp_df['R1'][half_rows:].mean()       

        self.assertEqual(avg_neg, expected_avg_neg)
        self.assertEqual(avg_pos, expected_avg_pos)
        

    def test_generate_interactions_df(self):
        dict_response={'Taste': [74, 81, 71, 42, 75, 77, 80, 32]}
        df_new, dict_R_effects, dict_R_avg_neg_and_pos=self.full_factorial.generate_interactions_df(dict_response)
        
        df_new=df_new.astype(int)

        df_check_interactions=pd.read_csv("unit_test_interactions_df.csv")
        expected_df_new=df_check_interactions.iloc[:-3,:].astype(int)

        #check df_new
        assert_frame_equal(df_new, expected_df_new)

        # check dict_R_effects
        expected_dict_effects=OrderedDict(df_check_interactions.iloc[8,1:-1])
        assert dict_R_effects['Taste']==expected_dict_effects

        # dict_R_avg_neg_and_pos
        dict_avg_neg=df_check_interactions.iloc[9,1:-1].to_dict()
        dict_avg_pos=df_check_interactions.iloc[10,1:-1].to_dict()
        
        
        expected_dict_R_avg_neg_and_pos = OrderedDict()

        # Iterate through the keys and create a new dictionary with tuples of values
        for key in dict_avg_neg.keys():
            expected_dict_R_avg_neg_and_pos[key] = (dict_avg_neg[key], dict_avg_pos[key])

        assert dict_R_avg_neg_and_pos['Taste']==expected_dict_R_avg_neg_and_pos

    
    def test_randomize_order(self):
        exp_df = self.full_factorial.generate_exp_df()
        randomized_exp_df=self.full_factorial.randomize_order(exp_df)

        assert not randomized_exp_df.equals(exp_df) 

if __name__ == '__main__':
    unittest.main()
