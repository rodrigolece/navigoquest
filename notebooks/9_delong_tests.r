# Install pROC only when it is missing. A bare install.packages() has no
# repository set, which errors in a non-interactive session (Rscript) and
# prompts for a CRAN mirror in an interactive one.
if (!requireNamespace("pROC", quietly = TRUE)) {
    install.packages("pROC", repos = "https://cloud.r-project.org")
}
library(pROC)

# Path relative to this script's directory (notebooks/), matching the notebooks
file_dir <- 'navigoquest/data/online/online_metric_roc_data'

key0 <- "voc"
key1 <- "duration"
key2 <- "vector_conformity"

levels   <- list(6, 8, 11)
quantiles <- list(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

output <- data.frame(
    level            = numeric(),
    quantile         = numeric(),
    auc1             = numeric(),
    auc1_se          = numeric(),
    auc2             = numeric(),
    auc2_se          = numeric(),
    pval_delong      = numeric(),
    std_auc_diff     = numeric()
)

# Main for-loop
for (lvl in levels) {
    for (quantile in quantiles) {

        # Load data
        filename <- sprintf("%s/roc_data_lvl%d_quantile%.1f.csv", file_dir, lvl, quantile)
        df <- read.csv(filename)

        # Access values
        y_true   <- df[[key0]]
        z_score1 <- df[[key1]]
        z_score2 <- df[[key2]]

        # ROC's
        roc1 <- roc(y_true, z_score1)
        roc2 <- roc(y_true, z_score2)

        # AUC with 95% CIs
        ci1 <- ci.auc(roc1)
        ci2 <- ci.auc(roc2)

        # DeLong test (paired, since same y_true)
        delong <- roc.test(roc1, roc2, method="delong", paired=TRUE)

        # Record values
        auc1         <- as.numeric(ci1[2])
        auc1_se      <- as.numeric((ci1[3] - ci1[1]) / (2 * 1.96))
        auc2         <- as.numeric(ci2[2])
        auc2_se      <- as.numeric((ci2[3] - ci2[1]) / (2 * 1.96))
        pval_delong  <- delong$p.value
        std_auc_diff <- as.numeric(delong$statistic)

        # Save to output
        output <- rbind(output, data.frame(
            level            = lvl,
            quantile         = quantile,
            auc1             = auc1,
            auc1_se          = auc1_se,
            auc2             = auc2,
            auc2_se          = auc2_se,
            pval_delong      = pval_delong,
            std_auc_diff     = std_auc_diff
        ))
    }
}

# Save output
output_name <- sprintf("%s/delong_results.csv", file_dir)
write.csv(output, output_name, row.names=FALSE)

print('Done.')


