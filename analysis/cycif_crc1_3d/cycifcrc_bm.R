# CyCIF CRC1 benchmark figures (ggplot classic style, base svg device)
# NTF-only benchmark (SpatialZ/FEAST infeasible at 12.8M cells — see
# baseline_feasibility/). Run from experiment/cycif_crc1_benchmark/ with the
# monocle3-r R environment.
library(tidyverse)

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

plot_line <- function(df, y_col, y_label, file_name, width = 8, height = 5.5) {
  df <- df %>% mutate(scenario = factor(scenario, levels = scen_levels))
  p <- ggplot(df, aes(x = scenario, y = .data[[y_col]], group = 1)) +
    geom_line(color = "#82659C", linewidth = 1) +
    geom_point(color = "#82659C", size = 3) +
    labs(x = "Scenario", y = y_label, title = paste("CyCIF CRC1 benchmark (NTF):", y_label)) +
    theme_bm
  save_svg(p, file_name, width, height)
}

# ==============================================================================
# 1. Reconstruction metrics
# ==============================================================================
recon <- read_csv("recon/recon_metrics_summary.csv", show_col_types = FALSE) %>%
  filter(method == "NTF")
plot_line(recon, "pearson_mean", "Pearson Correlation", "recon_pearson")
plot_line(recon, "spearman_mean", "Spearman Correlation", "recon_spearman")
plot_line(recon, "rmse_mean", "RMSE", "recon_rmse")
plot_line(recon, "ssim_3d_mean", "3D SSIM", "recon_ssim")

# ==============================================================================
# 2. SVG conservation: Moran's I corr + Top-K Jaccard
# ==============================================================================
svg_sum <- read_csv("svg/svg_benchmark_summary.csv", show_col_types = FALSE) %>%
  filter(method == "NTF") %>%
  rename(scenario = dataset)
plot_line(svg_sum, "Moran_I_corr", "Moran's I Correlation (GT vs pred)", "svg_moran_corr")

svg_long <- svg_sum %>%
  select(scenario, starts_with("SVG_Jaccard@")) %>%
  pivot_longer(starts_with("SVG_Jaccard@"), names_to = "K", values_to = "Jaccard",
               names_prefix = "SVG_Jaccard@") %>%
  mutate(K = factor(paste0("Top-", K), levels = c("Top-5", "Top-10", "Top-20")),
         scenario = factor(scenario, levels = scen_levels))
p_j <- ggplot(svg_long, aes(x = K, y = Jaccard, group = 1)) +
  geom_line(color = "#82659C", linewidth = 1) +
  geom_point(color = "#82659C", size = 3) +
  facet_wrap(~scenario, nrow = 1) +
  ylim(0, 1) +
  labs(x = "Top-K", y = "SVG Jaccard", title = "CyCIF CRC1 benchmark (NTF): Top-K SVG Jaccard") +
  theme_bm
save_svg(p_j, "svg_jaccard_topk", width = 13, height = 5)

# ==============================================================================
# 3. Cell typing + niche typing: ARI / NMI / FMI
# ==============================================================================
ct <- read_csv("celltyping/celltyping_metrics.csv", show_col_types = FALSE) %>%
  mutate(metric = factor(toupper(metric), levels = c("ARI", "NMI", "FMI")),
         scenario = factor(scenario, levels = scen_levels), level = "cell_type (21)")
nt <- read_csv("nichetyping/nichetyping_metrics.csv", show_col_types = FALSE) %>%
  filter(metric %in% c("ari", "nmi", "fmi")) %>%
  mutate(metric = factor(toupper(metric), levels = c("ARI", "NMI", "FMI")),
         scenario = factor(scenario, levels = scen_levels), level = "cell_type_main (niche, 9)")
both <- bind_rows(ct, nt)
p_ct <- ggplot(both, aes(x = scenario, y = value, fill = level)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  facet_wrap(~metric, nrow = 1) +
  ylim(0, 1) +
  labs(x = "Scenario", y = "Score", title = "CyCIF CRC1 benchmark (NTF): cell typing & niche typing accuracy",
       fill = "Annotation level") +
  scale_fill_manual(values = c("cell_type (21)" = "#82659C", "cell_type_main (niche, 9)" = "#4883B1")) +
  theme_bm + guides(fill = guide_legend(nrow = 1, byrow = TRUE))
save_svg(p_ct, "celltyping_nichetyping_ari_nmi_fmi", width = 12, height = 5.5)

# ==============================================================================
# 4. Niche composition fidelity
# ==============================================================================
nc <- read_csv("nichetyping/nichetyping_metrics.csv", show_col_types = FALSE) %>%
  filter(!metric %in% c("ari", "nmi", "fmi")) %>%
  mutate(scenario = factor(scenario, levels = scen_levels))
nc_cos <- nc %>% filter(metric %in% c("mean_cosine", "mean_js", "freq_pearson", "tv_distance")) %>%
  mutate(metric = factor(metric, levels = c("mean_cosine", "mean_js", "freq_pearson", "tv_distance")))
p_nc <- ggplot(nc_cos, aes(x = scenario, y = value, group = 1)) +
  geom_line(color = "#82659C", linewidth = 1) +
  geom_point(color = "#82659C", size = 3) +
  facet_wrap(~metric, nrow = 1, scales = "free_y") +
  labs(x = "Scenario", y = "Value",
       title = "CyCIF CRC1 benchmark (NTF): niche composition fidelity") +
  theme_bm
save_svg(p_nc, "niche_composition_metrics", width = 13, height = 5)

# per-niche cosine / JS (senario1 + senario2)
pn <- read_csv("nichetyping/niche_per_niche_metrics.csv", show_col_types = FALSE) %>%
  mutate(niche = fct_reorder(niche, cosine, .fun = median, .desc = TRUE))
p_pn <- ggplot(pn, aes(x = niche, y = cosine)) +
  geom_bar(stat = "identity", fill = "#82659C") +
  facet_wrap(~factor(scenario, levels = scen_levels), nrow = 1) +
  ylim(0, 1) +
  labs(x = "Niche (cell_type_main)", y = "Composition cosine similarity",
       title = "CyCIF CRC1 benchmark (NTF): per-niche composition fidelity") +
  theme_bm + theme(axis.text.x = element_text(angle = 45, hjust = 1))
save_svg(p_pn, "niche_per_niche_cosine", width = 12, height = 5.5)

# niche frequency scatter (pred vs GT)
comp_freq <- read_csv("nichetyping/niche_per_niche_metrics.csv", show_col_types = FALSE)
p_freq <- ggplot(comp_freq, aes(x = n_cells_gt, y = n_cells_pred)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  geom_point(color = "#82659C", size = 2.5) +
  facet_wrap(~factor(scenario, levels = scen_levels), nrow = 1) +
  scale_x_log10() + scale_y_log10() +
  labs(x = "GT niche size (cells)", y = "Predicted niche size (cells)",
       title = "CyCIF CRC1 benchmark (NTF): niche frequency pred vs GT") +
  theme_bm
save_svg(p_freq, "niche_size_scatter", width = 13, height = 5)

# niche composition stacked bar (senario1): GT vs NTF pred
comp <- read_csv("nichetyping/niche_composition_long.csv", show_col_types = FALSE) %>%
  filter(scenario == "senario1")
top_clusters <- comp %>% filter(source == "gt") %>%
  group_by(cluster) %>% summarise(total = sum(fraction)) %>%
  arrange(desc(total)) %>% slice_head(n = 12) %>% pull(cluster)
comp_plot <- comp %>%
  mutate(cluster_plot = ifelse(cluster %in% top_clusters, cluster, "Other"),
         source = factor(source, levels = c("gt", "pred"), labels = c("GT", "NTF pred")))
p_sb <- ggplot(comp_plot, aes(x = niche, y = fraction, fill = cluster_plot)) +
  geom_bar(stat = "identity", position = "stack") +
  facet_wrap(~source, nrow = 1) +
  labs(x = "Niche (cell_type_main)", y = "Cell-type composition (fraction)",
       title = "CyCIF CRC1 senario1: niche composition, GT vs NTF", fill = "cell_type") +
  theme_bm + theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  guides(fill = guide_legend(nrow = 2, byrow = TRUE))
save_svg(p_sb, "niche_composition_stackedbar_senario1", width = 12, height = 6)

# ==============================================================================
# 5. Baseline feasibility: FEAST scaling
# ==============================================================================
feast <- read_csv("baseline_feasibility/feast_scaling.csv", show_col_types = FALSE) %>%
  filter(exit_code == 0) %>%
  mutate(peak_rss_gb = peak_rss_kb / 1e6, peak_gpu_gb = peak_gpu_mem_mib / 1024)
# quadratic extrapolation to full scale (510k cells/slice)
fit_rss <- lm(log(peak_rss_gb) ~ log(n_per_slice) + I(log(n_per_slice)^2), data = feast)
fit_time <- lm(log(wall_time_s) ~ log(n_per_slice) + I(log(n_per_slice)^2), data = feast)
grid <- tibble(n_per_slice = 10^seq(log10(2000), log10(510000), length.out = 100))
grid$peak_rss_gb <- exp(predict(fit_rss, grid))
grid$wall_time_s <- exp(predict(fit_time, grid))
feast_long <- feast %>% select(n_per_slice, peak_rss_gb, wall_time_s) %>%
  pivot_longer(-n_per_slice, names_to = "quantity", values_to = "value")
grid_long <- grid %>% select(n_per_slice, peak_rss_gb, wall_time_s) %>%
  pivot_longer(-n_per_slice, names_to = "quantity", values_to = "value")
labels_q <- c(peak_rss_gb = "Peak RSS (GB)", wall_time_s = "Wall time per slice pair (s)")
p_fe <- ggplot() +
  geom_line(data = grid_long, aes(x = n_per_slice, y = value), linetype = "dashed", color = "grey40") +
  geom_point(data = feast_long, aes(x = n_per_slice, y = value), color = "#4883B1", size = 3) +
  geom_hline(data = data.frame(quantity = "peak_rss_gb", y = 1900),
             aes(yintercept = y), color = "red", linetype = "dotted") +
  geom_text(data = data.frame(quantity = "peak_rss_gb"),
            aes(x = 3000, y = 2600, label = "machine RAM 1.9 TB"), color = "red", hjust = 0, size = 3.5) +
  facet_wrap(~quantity, scales = "free_y", labeller = as_labeller(labels_q)) +
  scale_x_log10() + scale_y_log10() +
  labs(x = "Cells per slice (log scale)", y = NULL,
       title = "FEAST on CRC1: measured scaling (points) + quadratic extrapolation (dashed)",
       subtitle = "CRC1 full scale = 510k cells/slice; OT cost matrix alone at full scale = 1.04 TB (float32)") +
  theme_bm
save_svg(p_fe, "baseline_feasibility_scaling", width = 11, height = 5.5)

# ==============================================================================
# 6. Annotation QC
# ==============================================================================
gates <- read_csv("annotation/gating_thresholds.csv", show_col_types = FALSE) %>%
  mutate(marker = fct_reorder(marker, positive_fraction, .desc = TRUE))
p_gate <- ggplot(gates, aes(x = marker, y = positive_fraction, fill = gate_source)) +
  geom_bar(stat = "identity") +
  labs(x = "Marker", y = "Positive fraction",
       title = "CyCIF CRC1 annotation: per-marker positive fractions",
       fill = "Gate source") +
  scale_fill_manual(values = c("gmm" = "#82659C", "manual_adjusted" = "#D9A441")) +
  theme_bm + theme(axis.text.x = element_text(angle = 45, hjust = 1)) + guide_legend_row
save_svg(p_gate, "annotation_positive_fractions", width = 10, height = 5.5)

ctc <- read_csv("annotation/celltype_counts.csv", show_col_types = FALSE) %>%
  mutate(cell_type = fct_reorder(cell_type, n_cells, .fun = sum, .desc = TRUE))
p_ctc <- ggplot(ctc, aes(x = cell_type, y = n_cells, fill = cell_type_main)) +
  geom_bar(stat = "identity") +
  scale_y_log10() +
  labs(x = "cell_type", y = "Cells (log scale)",
       title = "CyCIF CRC1 annotation: cell-type composition", fill = "cell_type_main") +
  theme_bm + theme(axis.text.x = element_text(angle = 45, hjust = 1)) + guide_legend_row
save_svg(p_ctc, "annotation_celltype_counts", width = 12, height = 6)

# ==============================================================================
# 7. Efficiency
# ==============================================================================
eff <- read_csv("efficiency_summary.csv", show_col_types = FALSE) %>%
  mutate(scenario = factor(scenario, levels = scen_levels))
if ("train_time_s" %in% names(eff)) {
  p_eff <- ggplot(eff, aes(x = scenario, y = train_time_s / 60, group = 1)) +
    geom_line(color = "#82659C", linewidth = 1) + geom_point(color = "#82659C", size = 3) +
    labs(x = "Scenario", y = "Training time (min)", title = "CyCIF CRC1 benchmark (NTF): training time") +
    theme_bm
  save_svg(p_eff, "efficiency_time", width = 8, height = 5.5)
}
