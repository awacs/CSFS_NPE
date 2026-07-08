# corrdim_run.R
#
# Driver for corrDim4abc(): estimates the correlation dimension (D2) of the
# simulated CSFS cloud in ABC's own representation + metric, marks the ABC
# operating tolerance eps_abc, and reports the effective dimension.
#
# Mirrors abc_new.R: same normalization switch, same files, same metric --
# but measures the acceptance-count scaling exponent instead of running abc().
#
# Usage:
#   Rscript corrdim_run.R --target <csfs> --simTAG <tag> \
#       [--normalize median] [--distance euclidean] [--tol 0.01] \
#       [--nsub 4000] [--nrep 3] [--out corrdim]

library(devtools)
devtools::load_all("/u/home/h/haroldzw/abc")
library(argparse)

parser <- ArgumentParser(description = "Correlation dimension of simulated CSFS in ABC's metric")
parser$add_argument("--simTAG",    type = "character", help = "Simulation tag (expects <tag>.par.txt_DEN, <tag>.sim.txt_DEN)")
parser$add_argument("--normalize", type = "character", default = "median", help = "median | median_pre_cut | sum | sum_pre_cut | none (raw, 198 cells) | none_cut (raw, inner 158)")
parser$add_argument("--distance",  type = "character", default = "euclidean", help = "euclidean | manhattan | wasserstein")
parser$add_argument("--target",    type = "character", default = NULL, help = "OPTIONAL observed CSFS file; off by default. Only adds the eps_abc marker + pointwise acceptance curve")
parser$add_argument("--tol",       type = "double",    default = 0.01, help = "ABC acceptance rate for eps_abc (only used with --target)")
parser$add_argument("--nsub",      type = "integer",   default = 4000, help = "Subsample size for all-pairs integral")
parser$add_argument("--nrep",      type = "integer",   default = 3,    help = "Independent subsamples for robustness band")
parser$add_argument("--out",       type = "character", default = "corrdim", help = "Output basename for the figure")
args <- parser$parse_args()

CUT <- 10

## --- normalization, identical to abc_new.R -------------------------------
normalize_target <- switch(args$normalize,
  "median" = function(tgt, CUT) {
    v <- as.numeric(tgt)
    a <- (1+CUT):(99-CUT);  v[a] <- (v[a] - median(v[a])) / mad(v[a])
    b <- (100+CUT):(198-CUT); v[b] <- (v[b] - median(v[b])) / mad(v[b])
    v[c(a, b)]
  },
  "median_pre_cut" = function(tgt, CUT) {
    v <- as.numeric(tgt)
    a <- 1:99;   v[a] <- (v[a] - median(v[a])) / mad(v[a])
    b <- 100:198; v[b] <- (v[b] - median(v[b])) / mad(v[b])
    a <- (1+CUT):(99-CUT); b <- (100+CUT):(198-CUT)
    v[c(a, b)]
  },
  "sum" = function(tgt, CUT) {
    v <- as.numeric(tgt)
    a <- (1+CUT):(99-CUT);  v[a] <- v[a] / sum(v[a])
    b <- (100+CUT):(198-CUT); v[b] <- v[b] / sum(v[b])
    v[c(a, b)]
  },
  "sum_pre_cut" = function(tgt, CUT) {
    v <- as.numeric(tgt)
    a <- 1:99;   v[a] <- v[a] / sum(v[a])
    b <- 100:198; v[b] <- v[b] / sum(v[b])
    a <- (1+CUT):(99-CUT); b <- (100+CUT):(198-CUT)
    v[c(a, b)]
  },
  "none" = function(tgt, CUT) {        # raw CSFS, no transform, no cut (198 cells)
    as.numeric(tgt)
  },
  "none_cut" = function(tgt, CUT) {    # raw CSFS, no transform, cut to inner 158 cells
    v <- as.numeric(tgt)
    a <- (1+CUT):(99-CUT); b <- (100+CUT):(198-CUT)
    v[c(a, b)]
  },
  stop("unknown --normalize")
)

normalize_csfs_sim <- function(df, CUT)
  as.data.frame(t(apply(df, 1, normalize_target, CUT = CUT)))

## --- load data, same files as abc_new.R ----------------------------------
TAG      <- args$simTAG
csfs_sim <- read.table(paste0(TAG, ".sim.txt_DEN"), header = FALSE)
normalized_sim <- normalize_csfs_sim(csfs_sim, CUT)

## target is OPTIONAL (off by default) -- only loaded if --target is given
tgt <- if (!is.null(args$target))
           normalize_target(colSums(read.table(args$target)), CUT) else NULL

## --- correlation dimension in ABC's metric -------------------------------
cd <- corrDim4abc(sumstat  = normalized_sim,
                  target   = tgt,          # NULL unless --target supplied
                  distance = args$distance,
                  tol      = args$tol,
                  nsub     = args$nsub,
                  nrep     = args$nrep,
                  seed     = 1)

summary(cd)
plot(cd, file = args$out)
cat(sprintf("\nFigure written to %s.pdf\n", args$out))
