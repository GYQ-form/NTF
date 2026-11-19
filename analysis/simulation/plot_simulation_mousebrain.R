# Visualize simulation_mousebrain.xlsx (per-method sheets, metrics vs n_slices_used)
# monocle3-r env has no svglite, so plots are saved with the base svg device.
library(readxl)
library(tidyverse)

mycol <- c("SpatialZ"="#5FA24D", "NTF"="#82659C", "FEAST"="#4883B1", "UniST"="#D95F02")

# ==============================================================================
# 1. Read data (one sheet per method)
# ==============================================================================
excel_file_path <- "simulation_mousebrain.xlsx"
sheet_names <- excel_sheets(excel_file_path)

all_data <- map_dfr(sheet_names, function(sheet) {
  read_excel(excel_file_path, sheet = sheet, na = "NA") %>%
    mutate(Method = sheet)
})
all_data$Method <- factor(all_data$Method, levels = c("NTF", "FEAST", "SpatialZ", "UniST"))

# ==============================================================================
# 2. Plotting helper (compact panel style, as in the original plot.R)
# ==============================================================================
save_svg <- function(p, file_name, width = 6, height = 4) {
  svg(filename = file_name, width = width, height = height)
  print(p)
  dev.off()
  print(paste("Saved:", file_name))
}

plot_metric <- function(data, y_col) {
  ggplot(data, aes(x = n_slices_used, y = .data[[y_col]], col = Method)) +
    geom_line(linewidth = 1, na.rm = TRUE) +
    geom_point(size = 2, na.rm = TRUE) +
    labs(x = NULL, y = NULL) +
    scale_color_manual(values = mycol) +
    theme_classic() +
    theme(
      text = element_text(size = 14),
      plot.title = element_text(hjust = 0.5, face = "bold"),
      legend.position = "none",
      axis.text = element_text(size = 16),
      axis.line = element_line(color = "black"),
      axis.title = element_text(size = 18)
    )
}

# ==============================================================================
# 3. Metric plots
# ==============================================================================
save_svg(plot_metric(all_data, "avg_spearman"), "spearman.svg")
save_svg(plot_metric(all_data, "avg_ssim_3d"),  "ssim.svg")
save_svg(plot_metric(all_data, "avg_rmse"),     "rmse.svg")
