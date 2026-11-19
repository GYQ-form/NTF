# Per-scenario per-metric bar plots for wemerfish_benchmark_summary.xlsx (v1/refine)
# Same conventions as deep_starmap_benchmark/v3/refine/plot_per_scenario.R:
# compact bar + errorbar (recon only), no x label, no legend, base svg device;
# one figure per scenario x metric; NTF-only scenarios (senario3/head/tail)
# additionally get one all-metrics figure per scenario with adaptive ylim.
# Run: cd <v1/refine> && /home/gongyuqiao/.conda/envs/monocle3-r/bin/Rscript plot_per_scenario.R
library(readxl)
library(tidyverse)

mycol <- c("SpatialZ" = "#5FA24D", "NTF" = "#82659C", "FEAST" = "#4883B1", "UniST" = "#D95F02")
method_levels <- c("NTF", "FEAST", "SpatialZ", "UniST")
scenarios <- c("senario1", "senario2", "senario3", "head", "tail")
ntf_only <- c("senario3", "head", "tail")

excel_file_path <- "wemerfish_benchmark_summary.xlsx"
output_dir <- "./"

save_svg <- function(p, file_name, width = 3.5, height = 3) {
  svg(filename = file.path(output_dir, paste0(file_name, ".svg")),
      width = width, height = height)
  print(p)
  dev.off()
  print(paste("Saved:", file_name))
}

# One bar plot for a single scenario x metric; std_col NULL -> no errorbar
plot_scenario_metric <- function(data, scenario, mean_col, std_col, y_label) {
  d <- data %>%
    filter(scenario == !!scenario) %>%
    mutate(method = factor(method, levels = method_levels)) %>%
    arrange(method)
  if (nrow(d) == 0) return(invisible(NULL))
  p <- ggplot(d, aes(x = method, y = .data[[mean_col]], fill = method)) +
    geom_bar(stat = "identity", width = 0.6, color = "black", alpha = 0.8, na.rm = TRUE) +
    scale_fill_manual(values = mycol, drop = FALSE) +
    labs(x = NULL, y = y_label) +
    theme_classic() +
    theme(
      text = element_text(size = 14),
      plot.title = element_text(hjust = 0.5, face = "bold"),
      legend.position = "none",
      axis.text = element_text(size = 16),
      axis.text.x = element_text(angle = 45, hjust = 1),
      axis.line = element_line(color = "black"),
      axis.title = element_text(size = 18)
    )
  if (!is.null(std_col)) {
    p <- p + geom_errorbar(
      aes(ymin = pmax(0, .data[[mean_col]] - .data[[std_col]]),
          ymax = .data[[mean_col]] + .data[[std_col]]),
      width = 0.2, linewidth = 0.8, na.rm = TRUE)
  }
  save_svg(p, paste0("plot_", scenario, "_", y_label %>%
                       gsub(pattern = " |\\(|\\)|'", replacement = "_") %>%
                       gsub(pattern = "_+", replacement = "_") %>%
                       gsub(pattern = "_$", replacement = "")))
}

# ==============================================================================
# 1. Reconstruction metrics (spearman / RMSE / 3D SSIM, mean +/- cross-gene std)
# ==============================================================================
recon <- read_excel(excel_file_path, sheet = "recon_summary")
for (s in scenarios) {
  plot_scenario_metric(recon, s, "spearman_mean", "spearman_std", "Spearman")
  plot_scenario_metric(recon, s, "rmse_mean", "rmse_std", "RMSE")
  plot_scenario_metric(recon, s, "ssim_3d_mean", "ssim_3d_std", "3D_SSIM")
}

# ==============================================================================
# 2. SVG conservation metrics (Moran's I corr / SVG Jaccard, no std)
# ==============================================================================
svg_df <- read_excel(excel_file_path, sheet = "svg") %>%
  rename(scenario = dataset)
for (s in scenarios) {
  plot_scenario_metric(svg_df, s, "Moran_I_corr", NULL, "Morans_I_Corr")
  plot_scenario_metric(svg_df, s, "SVG_Jaccard", NULL, "SVG_Jaccard")
}

# ==============================================================================
# 3. Cell typing metrics (ARI / NMI / FMI, GT = clusters 51 classes; no std)
# ==============================================================================
ct <- read_excel(excel_file_path, sheet = "celltyping") %>%
  pivot_wider(names_from = metric, values_from = value) %>%
  rename(ARI = ari, NMI = nmi, FMI = fmi)
for (s in scenarios) {
  plot_scenario_metric(ct, s, "ARI", NULL, "ARI")
  plot_scenario_metric(ct, s, "NMI", NULL, "NMI")
  plot_scenario_metric(ct, s, "FMI", NULL, "FMI")
}

# ==============================================================================
# 4. Niche typing accuracy (ARI / NMI / FMI, GT = type 10 classes; no std)
# ==============================================================================
niche <- read_excel(excel_file_path, sheet = "nichetyping") %>%
  filter(composition_level == "na", metric %in% c("ari", "nmi", "fmi")) %>%
  pivot_wider(names_from = metric, values_from = value) %>%
  rename(ARI = ari, NMI = nmi, FMI = fmi)
for (s in scenarios) {
  plot_scenario_metric(niche, s, "ARI", NULL, "Niche_ARI")
  plot_scenario_metric(niche, s, "NMI", NULL, "Niche_NMI")
  plot_scenario_metric(niche, s, "FMI", NULL, "Niche_FMI")
}

# ==============================================================================
# 5. NTF-only scenarios (senario3/head/tail): one figure per scenario with all
#    eight metrics (recon pair / svg pair / celltyping pair / niche pair) as
#    group-colored bars; adaptive y-axis upper limit.
# ==============================================================================
group_col <- c("Reconstruction" = "#3F2A56", "SVG conservation" = "#5C4470",
               "Cell typing" = "#82659C", "Niche typing" = "#C7B3D4")
metric_order <- c("Spearman", "3D SSIM", "Moran's I Corr", "SVG Jaccard",
                  "ARI", "NMI", "Niche ARI", "Niche NMI")

build_ntf_only_long <- function(recon, svg_df, ct, niche) {
  r <- recon %>%
    filter(scenario %in% ntf_only, method == "NTF") %>%
    transmute(scenario,
              Spearman = spearman_mean, Spearman_std = spearman_std,
              `3D SSIM` = ssim_3d_mean, `3D SSIM_std` = ssim_3d_std)
  s <- svg_df %>%
    filter(scenario %in% ntf_only, method == "NTF") %>%
    transmute(scenario, `Moran's I Corr` = Moran_I_corr, `SVG Jaccard` = SVG_Jaccard)
  c <- ct %>%
    filter(scenario %in% ntf_only, method == "NTF") %>%
    transmute(scenario, ARI = ARI, NMI = NMI)
  n <- niche %>%
    filter(scenario %in% ntf_only, method == "NTF") %>%
    transmute(scenario, `Niche ARI` = ARI, `Niche NMI` = NMI)
  r %>% left_join(s, by = "scenario") %>% left_join(c, by = "scenario") %>%
    left_join(n, by = "scenario") %>%
    pivot_longer(cols = -scenario, names_to = "name", values_to = "value") %>%
    mutate(is_std = endsWith(name, "_std"),
           metric = sub("_std$", "", name)) %>%
    select(-name) %>%
    pivot_wider(names_from = is_std, values_from = value, names_prefix = "v") %>%
    rename(value = `vFALSE`, std = `vTRUE`) %>%
    mutate(
      group = case_when(metric %in% c("Spearman", "3D SSIM") ~ "Reconstruction",
                        metric %in% c("Moran's I Corr", "SVG Jaccard") ~ "SVG conservation",
                        metric %in% c("ARI", "NMI") ~ "Cell typing",
                        TRUE ~ "Niche typing"),
      group = factor(group, levels = names(group_col)),
      metric = factor(metric, levels = metric_order)
    )
}

ntf_only_long <- build_ntf_only_long(recon, svg_df, ct, niche)

for (s in ntf_only) {
  d <- ntf_only_long %>% filter(scenario == s) %>% arrange(metric)
  y_max <- max(d$value + ifelse(is.na(d$std), 0, d$std), na.rm = TRUE) * 1.15
  p <- ggplot(d, aes(x = metric, y = value, fill = group)) +
    geom_bar(stat = "identity", width = 0.6, color = "black", alpha = 0.8, na.rm = TRUE) +
    geom_errorbar(aes(ymin = pmax(0, value - std), ymax = value + std),
                  width = 0.2, linewidth = 0.8, na.rm = TRUE) +
    scale_fill_manual(values = group_col) +
    scale_y_continuous(limits = c(0, y_max), expand = expansion(mult = c(0, 0.02))) +
    labs(x = NULL, y = NULL) +
    theme_classic() +
    theme(
      text = element_text(size = 14),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 11),
      axis.text = element_text(size = 16),
      axis.text.x = element_text(angle = 45, hjust = 1),
      axis.line = element_line(color = "black"),
      axis.title = element_text(size = 18)
    )
  save_svg(p, paste0("plot_", s, "_allmetrics"), width = 5.8, height = 3.5)
}
