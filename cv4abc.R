library(devtools)
devtools::load_all("/u/home/h/haroldzw/abc")
library(argparse)

parser <- ArgumentParser(description = "Cross-validation / SBC for an ABC method (cv4abc)")
parser$add_argument("--rdata", type = "character",
                    default = "/u/scratch/h/haroldzw/CSFS/Simulation/C1/prufer_ghost_constant.csfswideprior.csv100000_4para_median.RData",
                    help = "Path to the .RData holding parsim, normalized_csfs_sim (and res)")
parser$add_argument("--method", type = "character", default = "neuralnet",
                    choices = c("rejection", "loclinear", "neuralnet", "ridge"),
                    help = "ABC method")
parser$add_argument("--transf", type = "character", default = "log",
                    choices = c("none", "log", "logit"),
                    help = "Parameter transformation (ignored for rejection)")
parser$add_argument("--no_hcorr", action = "store_true", default = FALSE,
                    help = "Disable the heteroscedastic correction (default: hcorr ON)")
parser$add_argument("--lambda", type = "character", default = "0.0001,0.001,0.01",
                    help = "Ridge/neuralnet penalty path, comma-separated (e.g. 0 or 0.1,1,10)")
parser$add_argument("--tol", type = "double", default = 0.05, help = "Acceptance tolerance")
parser$add_argument("--nval", type = "integer", default = 500, help = "Cross-validation sample size")
parser$add_argument("--ncores", type = "integer", default = 8, help = "Parallel cores")
parser$add_argument("--numnet", type = "integer", default = 50,
                    help = "Number of neural networks in the ensemble (neuralnet only, default: 50)")
args <- parser$parse_args()

# Extract everything from args BEFORE load(Rdata): the .RData (from abc_new.R's
# save.image) contains its own `args` object and will clobber this one.
Rdata      <- args$rdata
abc_method <- args$method
transf     <- args$transf
hcorr      <- !args$no_hcorr   # flag disables it; default TRUE
tol        <- args$tol
nval       <- args$nval
ncores     <- args$ncores
numnet     <- args$numnet
lambda     <- as.numeric(strsplit(args$lambda, ",")[[1]])
if (any(is.na(lambda))) stop("--lambda must be comma-separated numbers (e.g. 0 or 0.1,1,10)")

# Load the .RData into an isolated env and pull only the objects we need.
# The file comes from abc_new.R's save.image(), so it carries a whole workspace
# (its own `args`, etc.) that would otherwise clobber this script's variables.
dat <- new.env()
load(Rdata, envir = dat)
parsim              <- dat$parsim
normalized_csfs_sim <- dat$normalized_csfs_sim
res                 <- if (exists("res", envir = dat)) dat$res else NULL

qc_dir <- "/u/scratch/h/haroldzw/CSFS/ABC/QC/"
## transf/hcorr/lambda go in the prefix so ablation runs get distinct files and
## don't silently reuse a stale cached .rds from a different setting
## (lambda only matters for ridge/neuralnet, so it's tagged only there)
lambda_tag <- if (abc_method %in% c("ridge", "neuralnet"))
                paste0("_lambda", gsub(",", "-", args$lambda)) else ""
numnet_tag <- if (abc_method == "neuralnet") paste0("_numnet", numnet) else ""
out_prefix <- paste0(qc_dir, basename(Rdata), "_", abc_method,
                     "_transf", transf, "_hcorr", hcorr, lambda_tag, numnet_tag, "_")
rds_file <- paste0(out_prefix, "cv4abc.rds")

if (!file.exists(rds_file)) {
  cv_args <- list(param = parsim, sumstat = normalized_csfs_sim,
                  nval = nval, tol = tol, method = abc_method,
                  transf = transf, hcorr = hcorr, lambda = lambda, ncores = ncores)
  if (abc_method == "neuralnet") {
    cv_args <- c(cv_args, list(numnet = numnet, sizenet = 7, MaxNWts = 1500))
  }
  cv_res <- do.call(cv4abc, cv_args)
  saveRDS(cv_res, file = rds_file)
} else {
  cv_res <- readRDS(rds_file)
}

pdf(paste0(out_prefix, "cv4abc.pdf"))
plot(cv_res) # need to be run several times to get a good plot
dev.off()

# --- Robust error summary ---
pred  <- as.matrix(cv_res$estim[[names(cv_res$estim)[1]]])
truth <- as.matrix(cv_res$true)

rel_err <- sweep((pred - truth)^2, 2, apply(truth, 2, var) , "/")
robust_summary <- rbind(
  mean = apply(rel_err, 2, mean),
  median  = apply(rel_err, 2, median),
  trimmed = apply(rel_err, 2, mean, trim = 0.1),
  p90     = apply(rel_err, 2, quantile, 0.90)
)

capture.output(
  { cat("Default summary:\n"); print(summary(cv_res))
    cat("\nRobust summary:\n"); print(robust_summary) },
  file = paste0(out_prefix, "cv4abc.summary.txt")
)

print(summary(cv_res))
print(robust_summary)

pdf(paste0(out_prefix, "cv4abc_sbc.pdf"))
plot(cv_res, sbc = TRUE)
dev.off()

capture.output(
  summary(cv_res, sbc = TRUE),
  file = paste0(out_prefix, "cv4abc_sbc.txt")
)

if (!is.null(res)) {
  pdf(paste0(out_prefix, "abc_pairs.pdf"))
  pairs(res)
  dev.off()
}