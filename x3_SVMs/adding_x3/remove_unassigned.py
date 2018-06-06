import pandas as pd
import numpy as np
df = pd.read_csv('experimental_df_x3_pred.csv', sep = ',',skipinitialspace = False)
df = df.reset_index(drop = True)
df.replace(r'\s+', np.nan, regex=True)


nuclei_list = [  
            'Cys1_HA',
            'Cys2_HA',
            'Cys1_N',
            'Cys2_N',
            'Cys1_HN',
            'Cys2_HN',
            'Cys1_before_HN',
            'Cys2_before_HN',
            'Cys1_after_HN',
            'Cys2_after_HN',
            'Cys1_before_CA',
            'Cys2_before_CA',
            'Cys1_after_CA',
            'Cys2_after_CA',
            'Cys1_before_HA',
            'Cys2_before_HA',
            'Cys1_after_HA',
            'Cys2_after_HA']

print len(df)
for nuclei in nuclei_list:
  df[nuclei] = df[nuclei].replace(0,np.nan)
df = df.dropna(subset=[nuclei_list], thresh = len(nuclei_list) -3)
# print df[nuclei_list]
# print len(df)
# for nuclei in nuclei_list:
  # df[nuclei] = df[nuclei].replace(np.nan,df[nuclei].mean())
df = df.reset_index(drop = True)
print len(df)
df.to_csv('experimental_df_x3pred_refined.csv',index = False)