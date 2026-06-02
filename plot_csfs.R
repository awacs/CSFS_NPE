library(ggplot2)
library(patchwork)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) stop("Usage: Rscript plot_csfs.R <path/to/file.csfs> <left_pop> <right_pop> [y_max|free]")

infile   <- args[1]
left_pop <- args[2]
right_pop <- args[3]
y_max_arg <- if (length(args) >= 4) args[4] else "6000"

lines <- readLines(infile)
mat   <- do.call(rbind, lapply(lines, function(l) as.numeric(strsplit(l, "\t")[[1]])))
if (nrow(mat) > 1) {
  vals <- colSums(mat)
} else {
  vals <- as.numeric(mat[1, ])
}

n      <- length(vals)
if (n %% 2 != 0) stop("Input file must contain an even number of values, got ", n)
half_n <- n / 2

left_df  <- data.frame(index = 1:half_n, count = vals[1:half_n])
right_df <- data.frame(index = 1:half_n, count = vals[(half_n + 1):n])

y_max <- if (tolower(y_max_arg) %in% c("free", "auto", "none", "unlock")) NA else as.numeric(y_max_arg)

make_plot <- function(df, title) {
  p <- ggplot(df, aes(x = index, y = count)) +
    geom_point(color = "steelblue", size = 1.5) +
    labs(title = title, x = "Frequency bin", y = "Count") +
    scale_x_continuous(breaks = pretty(1:half_n)) +
    theme_bw()
  if (!is.na(y_max)) p <- p + ylim(0, y_max)
  p
}

p_left  <- make_plot(left_df,  paste(left_pop,  "CSFS"))
p_right <- make_plot(right_df, paste(right_pop, "CSFS"))

p_combined <- p_left + p_right

outfile <- paste0(tools::file_path_sans_ext(infile), "_plot.png")
ggsave(outfile, plot = p_combined, width = 8, height = 4, dpi = 150)
message("Saved to ", outfile)
