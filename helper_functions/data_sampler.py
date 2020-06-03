import pandas as pd
import sys
import csv
from helper_functions import *


tagOriginalDf = pd.read_csv('tagging-set-original_for_jupyter_tagging.csv')

to_brat = tagOriginalDf.sample(random_state = int(sys.argv[1]), 
                               n = int(sys.argv[2]))[[str('id'),'full_text']]

to_brat = cleaner(to_brat, text_col = 'full_text', is_pandas_series = False)

to_brat = to_brat.full_text.str.replace('\n','') 

to_brat = to_brat.apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))


to_brat.to_csv(sys.argv[3],
                            sep = ' ',
                            header = False,
                            index = False,
                            line_terminator = '\n\n',
                            quoting = csv.QUOTE_NONE, escapechar = ' ')
#iloc[0:35,1].
#'../brat-v1.3_Crunchy_Frog/data/test4.txt'
