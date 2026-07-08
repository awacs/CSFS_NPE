library(devtools)
devtools::load_all("/u/home/h/haroldzw/abc")

library(argparse)

# Create argument parser
parser <- ArgumentParser(description = "Script description")

# Add arguments
parser$add_argument("--target", type = "character", help = "Target argument")
parser$add_argument("--simTAG", type = "character", help = "SimTAG argument")
parser$add_argument("--normalize", type = "character", default = "median",
  help = "Type of normalization. Available options: 'median', 'median_pre_cut', 'sum', 'sum_pre_cut', 'none' (default: 'median')")
parser$add_argument("--threepara", action = "store_true", default = FALSE, help = "Threepara argument")
parser$add_argument("--distance", type = "character", default = "euclidean",
  help = "Distance metric for abc(). Available options: 'euclidean', 'manhattan', 'wasserstein' (default: 'euclidean')")
parser$add_argument("--method", type = "character", default = "neuralnet",
  help = "ABC method. Available options: 'rejection', 'loclinear', 'neuralnet', 'ridge' (default: 'neuralnet')")
parser$add_argument("--numnet", type = "integer", default = 50,
  help = "Number of neural networks in the ensemble (neuralnet only, default: 50)")
# Parse command line arguments
args <- parser$parse_args()

# Access the arguments
targetfile <- args$target
simTAG <- args$simTAG
threepara <- args$threepara
print(threepara)
distance_method <- args$distance
abc_method <- args$method
# Choose normalization method based on a variable
normalization_method <- args$normalize

valid_methods <- c("rejection", "loclinear", "neuralnet", "ridge")
if (!(abc_method %in% valid_methods)) {
  stop(paste0("Invalid method: '", abc_method, "'. Available options are: ",
              paste(valid_methods, collapse = ", "), "."))
}

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
  },
  "none" = function(tgt, CUT) {
    normalized_tgt <- as.numeric(tgt)
    first_range <- (1+CUT):(99-CUT)
    last_range <- (100+CUT):(198-CUT)
    normalized_tgt <- normalized_tgt[c(first_range, last_range)]
    return(normalized_tgt)
  }
)

if (is.null(normalize_target)) {
  stop(paste0(
    "Invalid normalization method: '", normalization_method, "'. ",
    "Available options are: 'median', 'median_pre_cut', 'sum', 'sum_pre_cut', 'none'."
  ))
}

#### PARAMETERS ###
TAG <- simTAG

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

# Name the parameter columns so cv4abc's accuracy/SBC plots carry real labels
# (matches PARAM_NAMES_DEFAULT in validate_npe.py). Read with header=F, so
# parsim otherwise keeps generic V1..V4 names. Done after the parsim$V2 uses
# above so those column references still resolve.
param_names <- c("alpha", "T_merge", "T_split", "Ghost_Ne")
colnames(parsim) <- param_names[1:ncol(parsim)]

abc_args <- list(target = tgt, param = parsim, sumstat = normalized_csfs_sim,
                 tol = 0.05, method = abc_method, distance = distance_method)
if (abc_method == "neuralnet") {
  abc_args <- c(abc_args, list(numnet = args$numnet, sizenet = 7, transf = "log", MaxNWts = 1500))
} else if (abc_method %in% c("ridge", "loclinear")) {
  abc_args <- c(abc_args, list(transf = "log"))
}
res <- do.call(abc, abc_args)
Rdata_dir <- "/u/scratch/h/haroldzw/CSFS/Pseudo_truth/RData_sim/"
## numnet only changes results for neuralnet; tag the .RData only there so
## ridge/loclinear/rejection cells keep stable, shareable filenames.
numnet_tag <- if (abc_method == "neuralnet") paste0("_numnet", args$numnet) else ""
save.image(paste0(Rdata_dir, basename(targetfile), basename(TAG),
                  ifelse(threepara, "_3para_", "_4para_"),
                  normalization_method, "_", distance_method, "_", abc_method,
                  numnet_tag, ".RData"))
