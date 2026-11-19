# Per-scenario all-metric bar plots for cycif_crc1_benchmark_summary.xlsx (v2/refine)
# NTF-only results: one figure per scenario; metrics grouped into
# Reconstruction (Spearman + 3D SSIM) / SVG conservation (Moran's I corr +
# SVG Jaccard@20) / Niche typing (ARI + NMI), three purple shades, adaptive
# y-axis upper limit. Mirrors deep_starmap_benchmark/v3/refine/plot_per_scenario.R
# section 4 (NTF-only scenarios). Base svg device (no svglite in monocle3-r).
# 运行: 在 experiment/cycif_crc1_benchmark/v2/ 目录下
#   Rscript refine/plot_per_scenario_ntf_only.R
library(readxl)
library(tidyverse)

xlsx_path <- "refine/cycif_crc1_benchmark_summary.xlsx"
output_dir <- "./refine/figures"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

scen_levels <- c("senario1", "senario2", "senario3", "head", "tail")

save_svg <- function(p, file_name, width = 4.5, height = 3.5) {
  svg(filename = file.path(output_dir, paste0(file_name, ".svg")),
      width = width, height = height)
  print(p)
  dev.off()
  print(paste("Saved:", file_name))
}

group_col <- c("Reconstruction" = "#5C4470", "SVG conservation" = "#82659C",
               "Niche typing" = "#C7B3D4")
metric_order <- c("Spearman", "3D SSIM", "Moran's I Corr", "SVG Jaccard", "ARI", "NMI")

recon <- read_excel(xlsx_path, sheet = "recon_summary")
svg_df <- read_excel(xlsx_path, sheet = "svg_summary") %>% rename(scenario = dataset)
niche <- read_excel(xlsx_path, sheet = "nichetyping") %>%
  filter(metric %in% c("ari", "nmi")) %>%
  pivot_wider(names_from = metric, values_from = value) %>%
  rename(ARI = ari, NMI = nmi)

all_long <- recon %>%
  transmute(scenario, Spearman = spearman_mean, `3D SSIM` = ssim_3d_mean) %>%
  left_join(svg_df %>% transmute(scenario, `Moran's I Corr` = Moran_I_corr,
                                 `SVG Jaccard` = .data[["SVG_Jaccard@20"]]),
            by = "scenario") %>%
  left_join(niche %>% transmute(scenario, ARI, NMI), by = "scenario") %>%
  pivot_longer(cols = -scenario, names_to = "metric", values_to = "value") %>%
  mutate(
    group = case_when(metric %in% c("Spearman", "3D SSIM") ~ "Reconstruction",
                      metric %in% c("Moran's I Corr", "SVG Jaccard") ~ "SVG conservation",
                      TRUE ~ "Niche typing"),
    group = factor(group, levels = c("Reconstruction", "SVG conservation", "Niche typing")),
    metric = factor(metric, levels = metric_order),
    scenario = factor(scenario, levels = scen_levels)
  )

for (s in scen_levels) {
  d <- all_long %>% filter(scenario == s) %>% arrange(metric)
  if (nrow(d) == 0) next
  y_max <- max(d$value, na.rm = TRUE) * 1.15
  p <- ggplot(d, aes(x = metric, y = value, fill = group)) +
    geom_bar(stat = "identity", width = 0.6, color = "black", alpha = 0.8) +
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
  save_svg(p, paste0("plot_", s, "_allmetrics"))
}

print("All per-scenario figures done.")
