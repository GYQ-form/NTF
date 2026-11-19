# 2x2 all-scale SVG benchmark plot, five methods (NTF/SpatialZ/FEAST/UniST-inr/UniST-fast).
# Mirrors the layout of svg_benchmark_allscales/benchmark_metrics_allscales.png
# (3D/2D x Moran's I corr / Top-20 SVG Jaccard; 2D panels carry a +-1 std ribbon
# across slices).  ggplot classic style; colors follow plot_r2t5_figures.R.

suppressPackageStartupMessages({
  library(ggplot2)
  library(gridExtra)
})

DIR <- "/mnt/experiments/gongyq/analysis_link/ur_annotation/NTF/mytrain/revision/experiment/r2t5_unist_benchmark/svg_scale"
df <- read.csv(file.path(DIR, "allmethods_svg_scale_metrics.csv"), stringsAsFactors = FALSE)

mycol <- c("NTF"="#82659C", "FEAST"="#4883B1", "SpatialZ"="#5FA24D",
           "UniST-inr"="#D95F02", "UniST-fast"="#F4A582")
mylevels <- c("NTF", "SpatialZ", "FEAST", "UniST-inr", "UniST-fast")
df$method <- factor(df$method, levels = mylevels)

base_theme <- theme_classic(base_size = 12) +
  theme(legend.position = "top", plot.title = element_text(size = 11))

make_panel <- function(d, ycol, stdcol, ylab) {
  d <- d[is.finite(d[[ycol]]), , drop = FALSE]
  if (!is.null(stdcol)) {
    d$ymin <- d[[ycol]] - d[[stdcol]]
    d$ymax <- d[[ycol]] + d[[stdcol]]
  }
  p <- ggplot(d, aes(x = scale_k, y = .data[[ycol]], color = method, group = method))
  if (!is.null(stdcol)) {
    p <- p + geom_ribbon(aes(ymin = ymin, ymax = ymax, fill = method),
                         alpha = 0.12, color = NA, na.rm = TRUE) +
      scale_fill_manual(values = mycol, guide = "none")
  }
  p + geom_line(linewidth = 0.9, na.rm = TRUE) +
    geom_point(size = 1.8, na.rm = TRUE) +
    scale_x_log10(breaks = c(100, 1000, 10000),
                  labels = c("10w", "100w", "1000w")) +
    scale_color_manual(values = mycol, drop = FALSE) +
    coord_cartesian(ylim = c(0, 1)) +
    base_theme +
    labs(x = "scale (*10^3 cells)", y = ylab, color = NULL, title = ylab)
}

p1 <- make_panel(df, "Moran_I_corr_3D", NULL, "Moran's I corr (3D)")
p2 <- make_panel(df, "SVG_Jaccard_3D", NULL, "SVG Jaccard Top-20 (3D)")
p3 <- make_panel(df, "Moran_I_corr_2D_mean", "Moran_I_corr_2D_std", "Moran's I corr (2D slices)")
p4 <- make_panel(df, "SVG_Jaccard_2D_mean", "SVG_Jaccard_2D_std", "SVG Jaccard Top-20 (2D slices)")

g <- arrangeGrob(p1, p2, p3, p4, ncol = 2)
svg(file.path(DIR, "benchmark_metrics_allscales_with_unist.svg"), width = 11, height = 8)
grid::grid.draw(g)
dev.off()
png(file.path(DIR, "benchmark_metrics_allscales_with_unist.png"), width = 11, height = 8,
    units = "in", res = 200)
grid::grid.draw(g)
dev.off()
cat("[written] benchmark_metrics_allscales_with_unist.{svg,png}\n")
