import pandas as pd

df          = pd.read_csv('experimental_df.csv',sep = ',', skipinitialspace = False)
# df_60 = df.loc[df['x1'] == (60)]
print df['Cys1_before_HN'].mean()
#df_m60 = df.loc[df['x1'] == (-60)]
#df_180 = df.loc[df['x1'] == (180)]

#df_60 = df.loc[df['cys1_ss_array'] == ("1,0.0,0.0")]
#df_m60 = df.loc[df['cys1_ss_array'] == ("0.0,1,0.0")]
#df_180 = df.loc[df['cys1_ss_array'] == ("0.0,0.0,1")]
#
#print 'all',df['cys1_phi'].mean()
#print '60',df_60['cys1_phi'].mean()
#print '-60',df_m60['cys1_phi'].mean()
#print '180',df_180['cys1_phi'].mean()
#
#
#print 'all',df['cys1_phi'].std()
#print '60',df_60['cys1_phi'].std()
#print '-60',df_m60['cys1_phi'].std()
#print '180',df_180['cys1_phi'].std()
## print df_60['cys1_before_psi']
##df_60.to_csv('60.csv')
#
#df_60 = df_60.fillna(df_60.mean())
#df_m60 = df_m60.fillna(df_m60.mean())
#df_180 = df_180.fillna(df_180.mean())
#
#df_split = df_60.append(df_m60, ignore_index = True)
#df_split = df_split.append(df_180, ignore_index = True)
#print (df_split)
#df_split.to_csv('DISH_updated.csv')