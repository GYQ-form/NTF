# Visualize simulation_allmetrics.xlsx (old metrics + new SVG metrics)
# Adapted from simulation_bm.R; scale column renamed, plots use base svg device
# (svglite is not installed in the monocle3-r env).
library(readxl)
library(tidyverse)

mycol <- c("SpatialZ"="#5FA24D", "NTF"="#82659C", "FEAST"="#4883B1", "UniST"="#D95F02")

# ==============================================================================
# 1. Config
# ==============================================================================
excel_file_path <- "simulation_scale_allmetrics.xlsx"
output_dir <- "./"

# ==============================================================================
# 2. Read data (one sheet per method)
# ==============================================================================
sheet_names <- excel_sheets(excel_file_path)

all_data <- map_dfr(sheet_names, function(sheet) {
  read_excel(excel_file_path, sheet = sheet, na = "NA") %>%
    rename(scale = `scale (*10^3)`) %>%
    select(-any_of("...10")) %>%
    mutate(Method = sheet) %>%
    mutate(scale = factor(scale, levels = unique(sort(scale))))
})
all_data$Method <- factor(all_data$Method, levels = c("NTF", "FEAST", "SpatialZ", "UniST"))

# ==============================================================================
# 3. Line-plot helper
# ==============================================================================
save_svg <- function(p, file_name, width = 10, height = 6) {
  svg(filename = file.path(output_dir, paste0(file_name, ".svg")),
      width = width, height = height)
  print(p)
  dev.off()
  print(paste("Saved:", file_name))
}

plot_line_metric <- function(data, y_col, y_label, file_name) {
  p <- ggplot(data, aes(x = scale, y = .data[[y_col]],
                        group = Method, color = Method)) +
    geom_line(linewidth = 1, na.rm = TRUE) +
    geom_point(size = 2, na.rm = TRUE) +
    labs(x = "Scale (Number of Cells, x10^3)",
         y = y_label,
         title = paste("Comparison of", y_label)) +
    scale_color_manual(values = mycol) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_classic() +
    theme(
      text = element_text(size = 14),
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      axis.line = element_line(color = "black")
    )
  save_svg(p, file_name)
}

# ==============================================================================
# 4. Line plots: original metrics
# ==============================================================================
plot_line_metric(all_data, "rmse_mean", "RMSE", "plot_RMSE")
plot_line_metric(all_data, "ssim_3d_mean", "3D SSIM", "plot_SSIM")
plot_line_metric(all_data, "spearman_corr_mean", "Spearman Correlation", "plot_Spearman")

# ==============================================================================
# 5. Line plots: new SVG metrics (3D and per-2D-slice)
# ==============================================================================
plot_line_metric(all_data, "Moran_I_corr_3D", "Moran's I Correlation (3D)", "plot_MoranI_3D")
plot_line_metric(all_data, "SVG_Jaccard_3D", "SVG Jaccard Top-20 (3D)", "plot_SVGJaccard_3D")
plot_line_metric(all_data, "Moran_I_corr_2D_mean", "Moran's I Correlation (2D slices)", "plot_MoranI_2D")
plot_line_metric(all_data, "SVG_Jaccard_2D_mean", "SVG Jaccard Top-20 (2D slices)", "plot_SVGJaccard_2D")

# ==============================================================================
# 6. Time cost bar plot (log10 scale)
# Manual numeric dodging: fixed bar width, each method keeps its own slot,
# scales without a method are left blank.
# ==============================================================================
n_methods <- nlevels(all_data$Method)
bar_width <- 0.18
time_data <- all_data %>%
  filter(!is.na(`time(min)`)) %>%
  mutate(x = as.numeric(scale) +
           (as.numeric(Method) - (n_methods + 1) / 2) * (bar_width + 0.02))

p_time <- ggplot(time_data, aes(x = x, y = `time(min)`, fill = Method)) +
  geom_col(width = bar_width) +
  scale_x_continuous(breaks = seq_along(levels(all_data$scale)),
                     labels = levels(all_data$scale)) +
  scale_y_log10(labels = scales::comma) +
  geom_hline(yintercept = 10, linetype = "dashed", color = "blue", linewidth = 0.8, alpha = 0.7) +
  geom_hline(yintercept = 1440, linetype = "dashed", color = "red", linewidth = 0.8, alpha = 0.7) +
  labs(x = "Scale (Number of Cells, x10^3)",
       y = "Time Cost (Minutes, log10 scale)",
       title = "Comparison of Time Consumption") +
  theme_classic() +
  theme(
    text = element_text(size = 14),
    legend.position = "bottom",
    axis.line = element_line(color = "black")
  ) +
  scale_fill_manual(values = mycol, drop = FALSE)

save_svg(p_time, "plot_TimeCost")
