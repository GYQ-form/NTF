# R2 Tech8: mouse embryo cell-type distribution figures (ggplot classic style, svg)
# Original (13 slices) vs NTF-reconstructed dense volume.
# Inputs (produced by compare_ct_distribution.py, run in this directory):
#   ct_distribution.csv, ct_distribution_metrics.csv, z_freq_long.csv
# Run (env: monocle3-r):
#   /home/gongyuqiao/.conda/envs/monocle3-r/bin/Rscript plot_distribution.R
library(tidyverse)

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

src_col <- c("original" = "grey40", "reconstructed" = "#82659C")

dist <- read_csv("ct_distribution.csv", show_col_types = FALSE)
metrics <- read_csv("ct_distribution_metrics.csv", show_col_types = FALSE)
mval <- function(m) metrics$value[metrics$metric == m][1]

# ==============================================================================
# 1. 30-class proportion bar chart (side by side, ordered by original proportion)
# ==============================================================================
dist_long <- dist %>%
  select(cell_type, original = prop_original, reconstructed = prop_reconstructed) %>%
  pivot_longer(-cell_type, names_to = "source", values_to = "prop") %>%
  mutate(source = factor(source, levels = c("original", "reconstructed")),
         cell_type = factor(cell_type, levels = rev(dist$cell_type)))
p_bar <- ggplot(dist_long, aes(x = cell_type, y = prop, fill = source)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = src_col) +
  labs(x = NULL, y = "Proportion of cells", fill = NULL,
       title = "Mouse embryo (E16.5): cell-type composition, original vs reconstructed") +
  theme_bm + guides(fill = guide_legend(nrow = 1, byrow = TRUE))
save_svg(p_bar, "ct_proportion_bar", width = 8.5, height = 8)

# ==============================================================================
# 2. Proportion scatter (log-log) with Pearson/Spearman annotation
# ==============================================================================
lab <- sprintf("Pearson r = %.3f\nSpearman rho = %.3f\nTV distance = %.3f",
               mval("pearson_r"), mval("spearman_r"), mval("tv_distance"))
# label the 10 representative organs that have 3D visualizations
rep_types <- c("Brain", "Spinal cord", "Eye", "Inner ear", "Heart",
               "Liver", "Lung", "Kidney", "GI tract", "Meninges")
p_sc <- ggplot(dist, aes(x = prop_original, y = prop_reconstructed)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  geom_point(size = 2.2, color = "#82659C") +
  geom_point(data = dist %>% filter(cell_type %in% rep_types),
             size = 2.6, color = "#D95F02") +
  ggrepel::geom_text_repel(
    data = dist %>% filter(cell_type %in% rep_types),
    aes(label = cell_type), size = 3.2, max.overlaps = 30,
    min.segment.length = 0.2, seed = 42) +
  annotate("text", x = min(dist$prop_original), y = max(dist$prop_reconstructed),
           label = lab, hjust = 0, vjust = 1, size = 4, color = "grey20") +
  scale_x_log10() + scale_y_log10() +
  labs(x = "Proportion in original (log scale)",
       y = "Proportion in reconstructed (log scale)",
       title = "Cell-type proportion concordance: original vs reconstructed",
       subtitle = "orange points: organs with 3D visualization") +
  theme_bm + theme(plot.subtitle = element_text(size = 10, color = "grey30"))
save_svg(p_sc, "ct_proportion_scatter", width = 7, height = 6)

# ==============================================================================
# 3. Per-z frequency curves: original 13 layers (points) vs reconstructed (line)
# ==============================================================================
zfreq <- read_csv("z_freq_long.csv", show_col_types = FALSE)
rep_types <- c("Brain", "Spinal cord", "Heart", "Liver",
               "Lung", "Kidney", "Eye", "GI tract")
zf <- zfreq %>%
  filter(cell_type %in% rep_types) %>%
  mutate(source = factor(source, levels = c("original", "reconstructed")),
         cell_type = factor(cell_type, levels = rep_types))
p_z <- ggplot(zf, aes(x = z_center, y = freq, color = source, group = source)) +
  geom_line(data = zf %>% filter(source == "reconstructed"), linewidth = 0.7) +
  geom_point(data = zf %>% filter(source == "original"), size = 1.8) +
  geom_line(data = zf %>% filter(source == "original"),
            linetype = "dashed", linewidth = 0.5) +
  facet_wrap(~cell_type, ncol = 4, scales = "free_y") +
  scale_color_manual(values = src_col) +
  labs(x = "z (scaled coordinate)", y = "Cell-type frequency", color = NULL,
       title = "Cell-type frequency along z: original 13 slices vs reconstructed dense volume",
       subtitle = "reconstructed fills the inter-slice gaps with a continuous z profile") +
  theme_bm + theme(plot.subtitle = element_text(size = 10, color = "grey30")) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE))
save_svg(p_z, "ct_freq_along_z", width = 12, height = 6)

print("All distribution figures done.")
