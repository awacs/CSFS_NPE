import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import dill
import argparse
import os
# the normalization doesn't work
def format_string(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        formatted_string = " + ".join([f"Q('{line.strip()}')" for line in lines])
        return formatted_string
def main(args):
  
  training_data = pd.read_csv(args.input, sep='\t', header=0)
  # training_data = pd.read_csv("archie2feature.txt", sep='\t', header=0)
  # X = training_data[['Min.Dist']] #change this
  # Xnorm = training_data[['Dist.Mean','Dist.Var','Dist.Skew','Dist.Kurtosis','Min.Dist','Num.Private',"Archaic"]]
  Xnorm = training_data
  # if args.normalize:
    # y = Xnorm[['Archaic']]
    # X=Xnorm
    # Xnorm=(X-X.mean())/X.std()
    # Xnorm[['Archaic']]=y
  
  formula="Archaic ~ "+format_string(args.features)
  print(formula)
  log_reg=smf.glm(formula=formula, data=Xnorm, family=sm.families.Binomial()).fit()
  print(log_reg.summary())
  log_reg = log_reg.remove_data()
  features_basename = os.path.basename(args.features)
  filehandler = open(args.input+features_basename+".obj","wb")
  dill.dump(log_reg,filehandler)
  ###
  # file = open(model,'rb')
  # log_reg = pickle.load(file)

# training_data['Prediction']=log_reg.predict(X)
# >>> out=training_data[['Haplotype.num','Chrom','Start','End','Prediction']]
# out.to_csv('out.txt',sep=" ",index=False)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input file")
    parser.add_argument("-f","--features", type=str, required=True, help="Features List")
    # parser.add_argument("-n","--normalize", action="store_true", required=False, help="Enable normalization mode")
    main(parser.parse_args()) 