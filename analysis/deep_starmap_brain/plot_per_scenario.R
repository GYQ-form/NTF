# Per-scenario per-metric bar plots for deep_starmap_benchmark.xlsx (v3/refine)
# Style follows bm.R (compact bar + errorbar, no x label, no legend);
# one figure per scenario x metric. Base svg device (no svglite in monocle3-r).
library(readxl)
library(tidyverse)

mycol <- c("SpatialZ" = "#5FA24D", "NTF" = "#82659C", "FEAST" = "#4883B1", "UniST" = "#D95F02")
method_levels <- c("NTF", "FEAST", "SpatialZ", "UniST")

excel_file_path <- "deep_starmap_benchmark.xlsx"
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
    geom_bar(stat = "identity", width = 0.6, color = "black", alpha = 0.8) +
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
      width = 0.2, linewidth = 0.8)
  }
  save_svg(p, paste0("plot_", scenario, "_", y_label %>%
                       gsub(pattern = " |\\(|\\)|'", replacement = "_") %>%
                       gsub(pattern = "_+", replacement = "_") %>%
                       gsub(pattern = "_$", replacement = "")))
}

# ==============================================================================
# 1. Reconstruction metrics (spearman / RMSE / 3D SSIM, mean +/- std)
# ==============================================================================
recon <- read_excel(excel_file_path, sheet = "recon_summary")
scenarios_recon <- unique(recon$scenario)
for (s in scenarios_recon) {
  plot_scenario_metric(recon, s, "spearman_mean", "spearman_std", "Spearman")
  plot_scenario_metric(recon, s, "rmse_mean", "rmse_std", "RMSE")
  plot_scenario_metric(recon, s, "ssim_3d_mean", "ssim_3d_std", "3D_SSIM")
}

# ==============================================================================
# 2. SVG conservation metrics (Moran's I corr / SVG Jaccard, no std)
# ==============================================================================
svg_df <- read_excel(excel_file_path, sheet = "svg_summary") %>%
  rename(scenario = dataset)
for (s in unique(svg_df$scenario)) {
  plot_scenario_metric(svg_df, s, "Moran_I_corr", NULL, "Morans_I_Corr")
  plot_scenario_metric(svg_df, s, "SVG_Jaccard", NULL, "SVG_Jaccard")
}

# ==============================================================================
# 3. Cell typing metrics (ARI / NMI / FMI, no std)
# ==============================================================================
ct <- read_excel(excel_file_path, sheet = "cell_typing_summary") %>%
  rename(scenario = dataset)
for (s in unique(ct$scenario)) {
  plot_scenario_metric(ct, s, "ARI", NULL, "ARI")
  plot_scenario_metric(ct, s, "NMI", NULL, "NMI")
  plot_scenario_metric(ct, s, "FMI", NULL, "FMI")
}

# ==============================================================================
# 4. NTF-only scenarios (senario3/4/5): one figure per scenario with all six
#    metrics (recon pair / svg pair / celltyping pair) as grouped-color bars;
#    adaptive y-axis upper limit.
# ==============================================================================
group_col <- c("Reconstruction" = "#5C4470", "SVG conservation" = "#82659C",
               "Cell typing" = "#C7B3D4")
scenario345 <- paste0("senario", 3:5)
metric_order <- c("Spearman", "3D SSIM", "Moran's I Corr", "SVG Jaccard", "ARI", "NMI")

build_ntf_only_long <- function(recon, svg_df, ct) {
  r <- recon %>%
    filter(scenario %in% scenario345) %>%
    transmute(scenario,
              Spearman = spearman_mean, Spearman_std = spearman_std,
              `3D SSIM` = ssim_3d_mean, `3D SSIM_std` = ssim_3d_std)
  s <- svg_df %>%
    filter(scenario %in% scenario345) %>%
    transmute(scenario, `Moran's I Corr` = Moran_I_corr, `SVG Jaccard` = SVG_Jaccard)
  c <- ct %>%
    filter(scenario %in% scenario345) %>%
    transmute(scenario, ARI = ARI, NMI = NMI)
  r %>% left_join(s, by = "scenario") %>% left_join(c, by = "scenario") %>%
    pivot_longer(cols = -scenario, names_to = "name", values_to = "value") %>%
    mutate(is_std = endsWith(name, "_std"),
           metric = sub("_std$", "", name)) %>%
    select(-name) %>%
    pivot_wider(names_from = is_std, values_from = value, names_prefix = "v") %>%
    rename(value = `vFALSE`, std = `vTRUE`) %>%
    mutate(
      group = case_when(metric %in% c("Spearman", "3D SSIM") ~ "Reconstruction",
                        metric %in% c("Moran's I Corr", "SVG Jaccard") ~ "SVG conservation",
                        TRUE ~ "Cell typing"),
      group = factor(group, levels = c("Reconstruction", "SVG conservation", "Cell typing")),
      metric = factor(metric, levels = metric_order)
    )
}

ntf_only <- build_ntf_only_long(recon, svg_df, ct)

for (i in 3:5) {
  s <- paste0("senario", i)
  d <- ntf_only %>% filter(scenario == s) %>% arrange(metric)
  y_max <- max(d$value + ifelse(is.na(d$std), 0, d$std), na.rm = TRUE) * 1.15
  p <- ggplot(d, aes(x = metric, y = value, fill = group)) +
    geom_bar(stat = "identity", width = 0.6, color = "black", alpha = 0.8) +
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
      axis.text = element_text(size = 16),
      axis.text.x = element_text(angle = 45, hjust = 1),
      axis.line = element_line(color = "black"),
      axis.title = element_text(size = 18)
    )
  save_svg(p, paste0("plot_", s, "_allmetrics"), width = 4.5, height = 3.5)
}
