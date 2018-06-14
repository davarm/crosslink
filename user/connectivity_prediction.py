import sys
sys.path.insert(0,'./modules')
import pandas as pd
from x3_prediction import crosslink_x3_prediction
from df_generation_connectivity import generate_connectivity
from crosslink_prediction import crosslink_prediction
peptide = sys.argv[1]

#----------------------------------
#----------------------------------
# Start with predicing X3 angles
#----------------------------------
#----------------------------------
df = pd.read_csv(peptide+'_crosslink.csv')
df = crosslink_x3_prediction(df)

#----------------------------------
#----------------------------------
# Generate the required inputs for connectivity
#----------------------------------
#----------------------------------
connectivity_df = generate_connectivity(df,peptide)

#----------------------------------
#----------------------------------
# Call neural network for prediction
#----------------------------------
#----------------------------------
crosslink_prediction(connectivity_df)