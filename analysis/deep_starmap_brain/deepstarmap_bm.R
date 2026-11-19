# Deep-STARmap benchmark figures (ggplot classic style, base svg device)
# Legends are always placed below the panel (single row) with an extra bottom
# margin, so they never overlap the plot body.
library(tidyverse)

mycol <- c("SpatialZ" = "#5FA24D", "NTF" = "#82659C", "FEAST" = "#4883B1")
scen_levels <- c("senario1", "senario2", "senario3", "head", "tail")
output_dir <- "./figures"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

save_svg <- function(p, file_name, width = 8, height = 5.5) {
  svg(filename = file.path(output_dir, paste0(file_name, ".svg")), width = width, height = height)
  print(p)
  dev.off()
  print(paste("Saved:", file_name))
}

theme_bm <- theme_classic() +
  theme(text = element_text(size = 13),
        legend.position = "bottom",
        legend.box = "horizontal",
        legend.margin = margin(t = 6),
        legend.box.margin = margin(t = 6),
        plot.margin = margin(10, 10, 16, 10),
        panel.grid.minor = element_blank(),
        axis.line = element_line(color = "black"))
guide_legend_row <- guides(fill = guide_legend(nrow = 1, byrow = TRUE))

prep <- function(df) {
  if ("dataset" %in% names(df)) df <- df %>% rename(scenario = dataset)
  df %>%
    mutate(method = factor(method, levels = c("NTF", "FEAST", "SpatialZ")),
           scenario = factor(scenario, levels = scen_levels))
}

plot_bar <- function(df, y_col, y_label, file_name, width = 8, height = 5.5) {
  p <- ggplot(df, aes(x = scenario, y = .data[[y_col]], fill = method)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
    labs(x = "Scenario", y = y_label, title = paste("Deep-STARmap benchmark:", y_label)) +
    scale_fill_manual(values = mycol, drop = FALSE) +
    theme_bm + guide_legend_row
  save_svg(p, file_name, width, height)
}

# ==============================================================================
# 1. Reconstruction metrics
# ==============================================================================
recon <- read_csv("recon/recon_metrics_summary.csv", show_col_types = FALSE) %>% prep()
plot_bar(recon, "pearson_mean", "Pearson Correlation", "recon_pearson")
plot_bar(recon, "spearman_mean", "Spearman Correlation", "recon_spearman")
plot_bar(recon, "rmse_mean", "RMSE", "recon_rmse")
plot_bar(recon, "ssim_3d_mean", "3D SSIM", "recon_ssim")

# ==============================================================================
# 2. SVG conservation: Moran's I corr + Top-K Jaccard
# ==============================================================================
svg_sum <- read_csv("svg/svg_benchmark_summary.csv", show_col_types = FALSE) %>% prep()
plot_bar(svg_sum, "Moran_I_corr", "Moran's I Correlation (GT vs pred)", "svg_moran_corr")

svg_long <- svg_sum %>%
  select(scenario, method, starts_with("SVG_Jaccard@")) %>%
  pivot_longer(starts_with("SVG_Jaccard@"), names_to = "K", values_to = "Jaccard",
               names_prefix = "SVG_Jaccard@") %>%
  mutate(K = factor(paste0("Top-", K), levels = c("Top-5", "Top-10", "Top-20", "Top-50")))
p_j <- ggplot(svg_long, aes(x = scenario, y = Jaccard, fill = method)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~K, nrow = 1) +
  labs(x = "Scenario", y = "SVG Jaccard", title = "Deep-STARmap benchmark: Top-K SVG Jaccard") +
  scale_fill_manual(values = mycol, drop = FALSE) +
  theme_bm + guide_legend_row
save_svg(p_j, "svg_jaccard_topk", width = 13, height = 5.5)

# ==============================================================================
# 3. Cell typing: ARI / NMI / FMI
# ==============================================================================
ct <- read_csv("celltyping/celltyping_metrics.csv", show_col_types = FALSE) %>%
  prep() %>%
  mutate(metric = factor(toupper(metric), levels = c("ARI", "NMI", "FMI")))
p_ct <- ggplot(ct, aes(x = scenario, y = value, fill = method)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~metric, nrow = 1) +
  labs(x = "Scenario", y = "Score", title = "Deep-STARmap benchmark: Cell typing (FEAST = clustering fallback)") +
  scale_fill_manual(values = mycol, drop = FALSE) +
  theme_bm + guide_legend_row
save_svg(p_ct, "celltyping_ari_nmi_fmi", width = 11, height = 5.5)

# ==============================================================================
# 4. Efficiency: train + inference time (log scale)
# ==============================================================================
eff <- read_csv("efficiency_summary.csv", show_col_types = FALSE) %>%
  prep() %>%
  mutate(train_time_s = ifelse(is.na(train_time_s), 0, train_time_s),
         total_time_s = train_time_s + infer_time_s)
p_eff <- ggplot(eff, aes(x = scenario, y = total_time_s, fill = method)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  scale_y_log10(labels = scales::comma) +
  labs(x = "Scenario", y = "Train + Inference Time (s, log10)",
       title = "Deep-STARmap benchmark: Computational efficiency (train+inference only)") +
  scale_fill_manual(values = mycol, drop = FALSE) +
  theme_bm + guide_legend_row
save_svg(p_eff, "efficiency_time", width = 8, height = 5.5)
