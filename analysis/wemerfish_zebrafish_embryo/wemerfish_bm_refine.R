# weMERFISH zebrafish embryo (C_6s_E2) benchmark figures — refine summary edition
# Data source: ./wemerfish_benchmark_summary.xlsx (refined summary tables)
#              ../nichetyping/niche_composition_long.csv + ../cluster_annotation_map.csv
#              (for the composition stacked bar, unchanged from v1)
# ggplot classic style, base svg device; legends always below the panel.
# Bar plots use position_dodge(preserve = "single") so scenarios where only NTF
# has results leave empty slots for the missing methods instead of stretching
# the NTF bar to the full width.
# Run: cd <v1/refine> && /home/gongyuqiao/.conda/envs/monocle3-r/bin/Rscript wemerfish_bm_refine.R
library(tidyverse)
library(readxl)

mycol <- c("SpatialZ" = "#5FA24D", "NTF" = "#82659C", "FEAST" = "#4883B1", "UniST" = "#D95F02")
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

# preserve = "single": bar width is computed as if all methods were present,
# leaving an empty slot for methods without results in a scenario
dodge_keep_gap <- position_dodge(width = 0.8, preserve = "single")

prep <- function(df) {
  if ("dataset" %in% names(df)) df <- df %>% rename(scenario = dataset)
  df %>%
    mutate(method = factor(method, levels = c("NTF", "FEAST", "SpatialZ", "UniST")),
           scenario = factor(scenario, levels = scen_levels))
}

plot_bar <- function(df, y_col, y_label, file_name, width = 8, height = 5.5) {
  p <- ggplot(df, aes(x = scenario, y = .data[[y_col]], fill = method)) +
    geom_bar(stat = "identity", position = dodge_keep_gap, width = 0.7) +
    labs(x = "Scenario", y = y_label, title = paste("weMERFISH zebrafish embryo:", y_label)) +
    scale_fill_manual(values = mycol, drop = FALSE) +
    theme_bm + guide_legend_row
  save_svg(p, file_name, width, height)
}

xlsx <- "wemerfish_benchmark_summary.xlsx"

# ==============================================================================
# 1. Reconstruction metrics
# ==============================================================================
recon <- read_excel(xlsx, sheet = "recon_summary") %>% prep()
plot_bar(recon, "spearman_mean", "Spearman Correlation", "recon_spearman")
plot_bar(recon, "rmse_mean", "RMSE", "recon_rmse")
plot_bar(recon, "ssim_3d_mean", "3D SSIM", "recon_ssim")

# ==============================================================================
# 2. SVG conservation: Moran's I corr + SVG Jaccard
# ==============================================================================
svg_df <- read_excel(xlsx, sheet = "svg") %>% prep()
plot_bar(svg_df, "Moran_I_corr", "Moran's I Correlation (GT vs pred)", "svg_moran_corr")
plot_bar(svg_df, "SVG_Jaccard", "SVG Jaccard", "svg_jaccard")

# ==============================================================================
# 3. Cell typing: ARI / NMI / FMI (GT = clusters, 51 classes)
# ==============================================================================
ct <- read_excel(xlsx, sheet = "celltyping") %>%
  prep() %>%
  mutate(metric = factor(toupper(metric), levels = c("ARI", "NMI", "FMI")))
p_ct <- ggplot(ct, aes(x = scenario, y = value, fill = method)) +
  geom_bar(stat = "identity", position = dodge_keep_gap, width = 0.7) +
  facet_wrap(~metric, nrow = 1) +
  labs(x = "Scenario", y = "Score",
       title = "weMERFISH: Cell typing vs GT clusters (FEAST = clustering fallback)") +
  scale_fill_manual(values = mycol, drop = FALSE) +
  theme_bm + guide_legend_row
save_svg(p_ct, "celltyping_ari_nmi_fmi", width = 11, height = 5.5)

# ==============================================================================
# 4. Niche typing accuracy: ARI / NMI / FMI (GT = type, 10 classes)
# ==============================================================================
niche_all <- read_excel(xlsx, sheet = "nichetyping") %>% prep()
niche_acc <- niche_all %>%
  filter(composition_level == "na", metric %in% c("ari", "nmi", "fmi")) %>%
  mutate(metric = factor(toupper(metric), levels = c("ARI", "NMI", "FMI")))
p_na <- ggplot(niche_acc, aes(x = scenario, y = value, fill = method)) +
  geom_bar(stat = "identity", position = dodge_keep_gap, width = 0.7) +
  facet_wrap(~metric, nrow = 1) +
  labs(x = "Scenario", y = "Score",
       title = "weMERFISH: Niche typing vs GT type (FEAST = leiden fallback, res=1.0)") +
  scale_fill_manual(values = mycol, drop = FALSE) +
  theme_bm + guide_legend_row
save_svg(p_na, "nichetyping_ari_nmi_fmi", width = 11, height = 5.5)

# ==============================================================================
# 4b. Niche composition fidelity summary (clusters level):
#     mean cosine / mean JS / niche-freq Pearson / TV distance
# ==============================================================================
comp_sum <- niche_all %>%
  filter(composition_level == "clusters",
         metric %in% c("mean_cosine", "mean_js", "freq_pearson", "tv_distance")) %>%
  mutate(metric = factor(metric,
                         levels = c("mean_cosine", "mean_js", "freq_pearson", "tv_distance"),
                         labels = c("Mean cosine", "Mean JS divergence",
                                    "Niche-freq Pearson", "TV distance")))
p_cs <- ggplot(comp_sum, aes(x = scenario, y = value, fill = method)) +
  geom_bar(stat = "identity", position = dodge_keep_gap, width = 0.7) +
  facet_wrap(~metric, nrow = 1) +
  labs(x = "Scenario", y = "Score",
       title = "weMERFISH: niche composition fidelity (pred vs GT, clusters level)") +
  scale_fill_manual(values = mycol, drop = FALSE) +
  theme_bm + guide_legend_row
save_svg(p_cs, "niche_composition_metrics", width = 13, height = 5.5)

# ==============================================================================
# 5. Niche composition stacked bar (core figure; source tables unchanged from v1)
# ==============================================================================
comp <- read_csv("../nichetyping/niche_composition_long.csv", show_col_types = FALSE) %>%
  filter(composition_level == "clusters",
         !startsWith(niche, "unmatched::")) %>%
  mutate(method = factor(method, levels = c("GT", "NTF", "SpatialZ", "FEAST", "UniST")))

cluster_map <- read_csv("../cluster_annotation_map.csv", show_col_types = FALSE)
keep_clusters <- cluster_map %>% arrange(desc(n_cells)) %>% head(12) %>% pull(cluster)

for (sc in c("senario1", "senario2")) {
  d <- comp %>% filter(scenario == sc)
  gt <- d %>% filter(method == "GT")
  niche_ord <- gt %>% group_by(niche) %>% summarise(f = sum(fraction), .groups = "drop") %>%
    arrange(desc(f)) %>% pull(niche)
  d2 <- d %>%
    mutate(cluster_plot = ifelse(cluster %in% keep_clusters, cluster, "Other"),
           niche = factor(niche, levels = niche_ord),
           cluster_plot = factor(cluster_plot, levels = c(keep_clusters, "Other")))
  pal <- c(scales::hue_pal()(12), "grey70")
  names(pal) <- c(keep_clusters, "Other")
  p_comp <- ggplot(d2, aes(x = niche, y = fraction, fill = cluster_plot)) +
    geom_bar(stat = "identity", width = 0.75) +
    facet_wrap(~method, nrow = 1) +
    labs(x = NULL, y = "Fraction of cells",
         title = paste0("weMERFISH ", sc, ": niche composition over GT clusters"),
         subtitle = "FEAST: Hungarian-matched leiden clusters only (unmatched excluded); top-12 clusters by cell count shown, rest pooled as Other") +
    scale_fill_manual(values = pal, name = "Cluster") +
    theme_bm +
    theme(axis.text.x = element_text(angle = 35, hjust = 1, size = 9)) +
    guides(fill = guide_legend(nrow = 4, byrow = TRUE))
  save_svg(p_comp, paste0("niche_composition_stackedbar_", sc), width = 14, height = 6.5)
}

# ==============================================================================
# 6. Per-niche JS divergence / cosine similarity (senario1/2, all methods)
# ==============================================================================
pn <- read_excel(xlsx, sheet = "niche_per_niche") %>%
  prep() %>%
  filter(composition_level == "clusters", scenario %in% c("senario1", "senario2")) %>%
  select(scenario, method, niche, cosine, js_divergence) %>%
  pivot_longer(c(cosine, js_divergence), names_to = "metric", values_to = "value") %>%
  mutate(metric = factor(metric, levels = c("cosine", "js_divergence"),
                         labels = c("Cosine similarity", "JS divergence")))
pn$niche <- factor(pn$niche, levels = rev(sort(unique(pn$niche))))
p_pn <- ggplot(pn, aes(y = niche, x = value, fill = method)) +
  geom_bar(stat = "identity", position = dodge_keep_gap, width = 0.7) +
  facet_grid(metric ~ scenario, scales = "free_x") +
  labs(y = NULL, x = "Score",
       title = "weMERFISH: per-niche composition fidelity (pred vs GT)") +
  scale_fill_manual(values = mycol, drop = FALSE) +
  theme_bm + guide_legend_row
save_svg(p_pn, "niche_js_cosine", width = 13, height = 7)

# ==============================================================================
# 7. Niche size scatter: predicted vs GT niche frequency (all scenarios)
# ==============================================================================
pn_all <- read_excel(xlsx, sheet = "niche_per_niche") %>%
  prep() %>%
  filter(composition_level == "clusters", !startsWith(niche, "unmatched::")) %>%
  group_by(scenario, method) %>%
  mutate(gt_freq = n_cells_gt / sum(n_cells_gt),
         pred_freq = n_cells_pred / sum(n_cells_pred)) %>%
  ungroup()
p_sc <- ggplot(pn_all, aes(x = gt_freq, y = pred_freq, color = method)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  geom_point(size = 2.5, alpha = 0.85) +
  facet_wrap(~scenario, nrow = 2) +
  labs(x = "GT niche frequency", y = "Predicted niche frequency",
       title = "weMERFISH: niche size distribution, predicted vs GT") +
  scale_color_manual(values = mycol, drop = FALSE) +
  theme_bm +
  guides(color = guide_legend(nrow = 1, byrow = TRUE))
save_svg(p_sc, "niche_size_scatter", width = 10, height = 7.5)

# ==============================================================================
# 8. Efficiency: train + inference time (SpatialZ niche re-runs excluded)
# ==============================================================================
eff <- read_excel(xlsx, sheet = "efficiency") %>%
  filter(!grepl("_niche", scenario)) %>%
  prep() %>%
  mutate(train_time_s = ifelse(is.na(train_time_s), 0, train_time_s),
         total_time_s = train_time_s + infer_time_s)
p_eff <- ggplot(eff, aes(x = scenario, y = total_time_s, fill = method)) +
  geom_bar(stat = "identity", position = dodge_keep_gap, width = 0.7) +
  scale_y_log10(labels = scales::comma) +
  labs(x = "Scenario", y = "Train + Inference Time (s, log10)",
       title = "weMERFISH: computational efficiency (train+inference only)") +
  scale_fill_manual(values = mycol, drop = FALSE) +
  theme_bm + guide_legend_row
save_svg(p_eff, "efficiency_time", width = 8, height = 5.5)
