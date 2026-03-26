# library(abc)
library(devtools)
devtools::load_all("/u/home/h/haroldzw/abc")
library(argparse)

# Create argument parser
parser <- ArgumentParser(description = "Script description")

# Add arguments
parser$add_argument("--target", type = "character", help = "The data being ABD'd")
parser$add_argument("--simTAG", type = "character", help = "The simulation tag specifing the simulation file")
parser$add_argument("--normalize", type = "character", default = "median",help = "type of normalization")
parser$add_argument("--threepara", action = "store_true", default = FALSE, help = "Threepara argument")
# Parse command line arguments
args <- parser$parse_args()

# Access the arguments
targetfile <- args$target
simTAG <- args$simTAG
threepara <- args$threepara
print(threepara)
# Choose normalization method based on a variable
normalization_method <- args$normalize

normalize_target <- switch(normalization_method,
  "median" = function(tgt, CUT) {
    normalized_tgt <- as.numeric(tgt)
    
    first_range <- (1+CUT):(99-CUT)
    median_first_range <- median(normalized_tgt[first_range])
    mad_first_range <- mad(normalized_tgt[first_range])
    normalized_tgt[first_range] <- (normalized_tgt[first_range] - median_first_range) / mad_first_range
    
    last_range <- (100+CUT):(198-CUT)
    median_last_range <- median(normalized_tgt[last_range])
    mad_last_range <- mad(normalized_tgt[last_range])
    normalized_tgt[last_range] <- (normalized_tgt[last_range] - median_last_range) / mad_last_range
    
    normalized_tgt <- normalized_tgt[c(first_range,last_range)]
    return(normalized_tgt)
  },
  "median_pre_cut" = function(tgt, CUT) {
    normalized_tgt <- as.numeric(tgt)
    
    first_range <- 1:99
    median_first_range <- median(normalized_tgt[first_range])
    mad_first_range <- mad(normalized_tgt[first_range])
    normalized_tgt[first_range] <- (normalized_tgt[first_range] - median_first_range) / mad_first_range
    
    last_range <- 100:198
    median_last_range <- median(normalized_tgt[last_range])
    mad_last_range <- mad(normalized_tgt[last_range])
    normalized_tgt[last_range] <- (normalized_tgt[last_range] - median_last_range) / mad_last_range
    
    first_range <- (1+CUT):(99-CUT)
    last_range <- (100+CUT):(198-CUT)
    normalized_tgt <- normalized_tgt[c(first_range,last_range)]
    return(normalized_tgt)
  },
  "sum" = function(tgt, CUT) {
    normalized_tgt <- as.numeric(tgt)
    
    first_range <- (1+CUT):(99-CUT)
    normalized_tgt[first_range] <- (normalized_tgt[first_range]) / sum(normalized_tgt[first_range])
    
    last_range <- (100+CUT):(198-CUT)
    normalized_tgt[last_range] <- (normalized_tgt[last_range]) / sum(normalized_tgt[last_range])
    
    normalized_tgt <- normalized_tgt[c(first_range,last_range)]
    return(normalized_tgt)
  },
  "sum_pre_cut" = function(tgt, CUT) {
    normalized_tgt <- as.numeric(tgt)
    
    first_range <- 1:99
    normalized_tgt[first_range] <- (normalized_tgt[first_range]) / sum(normalized_tgt[first_range])
    
    last_range <- 100:198
    normalized_tgt[last_range] <- (normalized_tgt[last_range]) / sum(normalized_tgt[last_range])
    
    first_range <- (1+CUT):(99-CUT)
    last_range <- (100+CUT):(198-CUT)
    normalized_tgt <- normalized_tgt[c(first_range,last_range)]
    return(normalized_tgt)
  }
)


#### PARAMETERS ###
TAG=simTAG

CUT=10



normalize_csfs_sim <- function(df, CUT) {
  
  normalized_df <- t(apply(df, 1, normalize_target, CUT = CUT))
  normalized_df <- as.data.frame(normalized_df)
  
  return(normalized_df)
}


# parfile=paste0(TAG,".par.txt")
# csfsfile=paste0(TAG,".sim.txt")
parfile=paste0(TAG,".par.txt_DEN")
csfsfile=paste0(TAG,".sim.txt_DEN")


parsim=read.table(parfile,header=F)
csfs_sim <- read.table(csfsfile,header=F)
line_count_parfile <- nrow(parsim)
line_count_csfsfile <- nrow(csfs_sim)

line_count_parfile == line_count_csfsfile


target=read.table(targetfile)
target=colSums(target)
tgt=normalize_target(target,CUT)
normalized_csfs_sim=normalize_csfs_sim(csfs_sim,CUT)
if (threepara){
  parsim=parsim[,1:3]
}


# for(j in 1:nss){
    # target[j] <- normalise(target[j],sumstat[,j])
# }
l=parsim$V2>2155   # pre OOA
m=parsim$V2<2155   # post OOA
res <- abc(target=tgt, param=parsim,sumstat=normalized_csfs_sim, tol=0.01,method="neuralnet",numnet=50,sizenet=7,transf="log",MaxNWts = 1500)

save.image(paste0(targetfile, TAG, ifelse(threepara, "_3para_", "_4para_"), normalization_method, ".RData"))

