# R2.t5 UniST benchmark figures (ggplot classic, svg + png, legend bottom).
# Inputs: scale_benchmark_summary_4methods.csv / interval_4methods_summary.csv /
#         gene_count sheets / per-benchmark summaries / efficiency_summary.csv
# Output: figures/fig2a_*.svg, fig2d_*.svg, fig_gene_count_*.svg, fig_small_benchmarks.svg,
#         fig_efficiency_scale.svg (+ .png each)
library(tidyverse)

EXP <- "/mnt/experiments/gongyq/analysis_link/ur_annotation/NTF/mytrain/revision/experiment/r2t5_unist_benchmark"
OUT <- file.path(EXP, "figures")
dir.create(OUT, showWarnings = FALSE)

mycol <- c("NTF"="#82659C", "FEAST"="#4883B1", "SpatialZ"="#5FA24D",
           "UniST-inr"="#D95F02", "UniST-fast"="#F4A582")
mylevels <- c("NTF", "SpatialZ", "FEAST", "UniST-inr", "UniST-fast")

save_both <- function(p, name, w=8, h=5.5) {
  svg(file.path(OUT, paste0(name, ".svg")), width=w, height=h); print(p); dev.off()
  ggsave(file.path(OUT, paste0(name, ".png")), p, width=w, height=h, dpi=150)
  cat("saved", name, "\n")
}

base_theme <- theme_classic() +
  theme(text=element_text(size=13), legend.position="bottom",
        axis.line=element_line(color="black"))

## ================= Fig 2a: scale benchmark =================
sc <- read.csv(file.path(EXP, "scale_benchmark_summary_4methods.csv")) %>%
  mutate(method=factor(method, levels=mylevels))
for (m in c("pearson_corr_mean","spearman_corr_mean","rmse_mean","ssim_3d_mean")) {
  ylab <- c(pearson_corr_mean="Pearson", spearman_corr_mean="Spearman",
            rmse_mean="RMSE", ssim_3d_mean="3D SSIM")[[m]]
  p <- ggplot(sc, aes(x=scale_num, y=.data[[m]], color=method, group=method)) +
    geom_line(linewidth=0.9, na.rm=TRUE) + geom_point(size=1.8, na.rm=TRUE) +
    scale_x_log10(breaks=c(10,100,1000), labels=c("10w","100w","1000w")) +
    scale_color_manual(values=mycol) + base_theme +
    labs(x="Scale (cells)", y=ylab, color=NULL,
         title=paste0("Scale benchmark: ", ylab))
  save_both(p, paste0("fig2a_", gsub("_mean","",m)), w=8, h=5.5)
}

## runtime panel (log-log): NTF train time; FEAST/SpatialZ inference; UniST serial_estimate
eff <- read.csv(file.path(EXP, "efficiency_summary.csv"))
eff_sc <- eff %>% filter(benchmark=="scale_test") %>%
  mutate(time_s = case_when(method=="NTF" ~ train_time_s,
                            method=="UniST-inr" ~ serial_estimate_s,
                            method=="UniST-fast" ~ serial_estimate_s,
                            TRUE ~ inference_time_s),
         scale_num = as.numeric(gsub("w","",scale))) %>%
  filter(!is.na(time_s)) %>%
  mutate(method=factor(method, levels=mylevels))
p <- ggplot(eff_sc, aes(x=scale_num, y=time_s, color=method, group=method)) +
  geom_line(linewidth=0.9, na.rm=TRUE) + geom_point(size=1.8, na.rm=TRUE) +
  scale_x_log10(breaks=c(10,100,1000), labels=c("10w","100w","1000w")) +
  scale_y_log10() +
  scale_color_manual(values=mycol) + base_theme +
  labs(x="Scale (cells)", y="Time (s, log; UniST=serial estimate)", color=NULL,
       title="Scale benchmark: runtime")
save_both(p, "fig2a_runtime", w=8, h=5.5)

## ================= Fig 2d: interval =================
itv <- read.csv(file.path(EXP, "eval/interval/interval_4methods_summary.csv")) %>%
  filter(k > 1) %>%  # k=1 is the trivial all-train baseline (prediction=GT by convention)
  mutate(n_train_slices = round(100 / k),
         method=factor(method, levels=mylevels))
for (m in c("avg_pearson","avg_spearman","avg_mse","avg_ssim_3d")) {
  ylab <- c(avg_pearson="Pearson", avg_spearman="Spearman",
            avg_mse="RMSE", avg_ssim_3d="3D SSIM")[[m]]
  d <- itv
  note <- NULL
  if (m=="avg_ssim_3d") {
    note <- "SpatialZ SSIM at k=5..50: NA (see README cleanup incident)"
  }
  p <- ggplot(d, aes(x=n_train_slices, y=.data[[m]], color=method, group=method)) +
    geom_line(linewidth=0.9, na.rm=TRUE) + geom_point(size=1.8, na.rm=TRUE) +
    scale_color_manual(values=mycol) + base_theme +
    labs(x="Number of input (train) slices", y=ylab, color=NULL,
         title=paste0("Sparse-slice benchmark: ", ylab),
         subtitle=note) +
    theme(plot.subtitle=element_text(size=10, color="#B22222"))
  save_both(p, paste0("fig2d_", gsub("avg_","",m)), w=8, h=5.5)
}

## ================= gene_count =================
gc3 <- read.csv("/mnt/experiments/gongyq/analysis_link/ur_annotation/NTF/mytrain/revision/experiment/gene_count_benchmark/summary_by_genes.csv") %>%
  select(method, n_genes, pearson_mean, spearman_mean, rmse_mean, ssim_3d_mean)
gcu <- read.csv(file.path(EXP, "eval/gene_count/gene_count_unist_summary.csv")) %>%
  mutate(method=paste0("UniST-", mode)) %>%
  select(method, n_genes, pearson_mean, spearman_mean, rmse_mean)
gcall <- bind_rows(gc3, gcu) %>% mutate(method=factor(method, levels=mylevels))
p <- ggplot(gcall, aes(x=n_genes, y=pearson_mean, color=method, group=method)) +
  geom_line(linewidth=0.9, na.rm=TRUE) + geom_point(size=2, na.rm=TRUE) +
  scale_x_log10(breaks=c(1,10,100,1000,10000)) +
  scale_color_manual(values=mycol) + base_theme +
  labs(x="Number of genes", y="Pearson", color=NULL,
       title="Gene-count benchmark (1M cells)",
       subtitle="FEAST 1-gene NA (method limit); UniST-inr 1-gene NA (official masked-MAE crash)") +
  theme(plot.subtitle=element_text(size=10, color="#B22222"))
save_both(p, "fig_gene_count_pearson", w=8, h=5.5)

## ================= small benchmarks (bar) =================
mk_bar <- function(df, xcol, ycol, title, fname, ylabel="Pearson") {
  df$method <- factor(df$method, levels=mylevels)
  p <- ggplot(df, aes(x=.data[[xcol]], y=.data[[ycol]], fill=method)) +
    geom_col(position=position_dodge(0.8), width=0.7) +
    scale_fill_manual(values=mycol) + base_theme +
    labs(x=NULL, y=ylabel, fill=NULL, title=title)
  save_both(p, fname, w=8, h=5.5)
}

mv <- read.csv(file.path(EXP, "eval/mousevc/mousevc_4methods_summary.csv"))
mk_bar(mv, "scenario", "pearson_mean", "STARmap mouseVC 5-gene (vs observed X)",
       "fig_mousevc_pearson")

dsr <- read.csv(file.path(EXP, "eval/deepstarmap/recon/recon_metrics_summary.csv")) %>%
  filter(scenario %in% c("senario1","senario2")) %>%
  mutate(method=ifelse(method=="UniST","UniST-inr",method))
mk_bar(dsr, "scenario", "pearson_mean", "Deep-STARmap brain v3 (senario1/2)",
       "fig_deepstarmap_pearson")

wmr <- read.csv(file.path(EXP, "eval/wemerfish/recon/recon_metrics_summary.csv")) %>%
  filter(scenario %in% c("senario1","senario2")) %>%
  mutate(method=ifelse(method=="UniST","UniST-inr",method))
mk_bar(wmr, "scenario", "pearson_mean", "weMERFISH (senario1/2)",
       "fig_wemerfish_pearson")

cy <- bind_rows(
  read.csv("/mnt/experiments/gongyq/analysis_link/ur_annotation/NTF/mytrain/revision/experiment/cycif_crc1_benchmark/v2/recon/recon_metrics_summary.csv") %>%
    filter(scenario %in% c("senario1","senario2")) %>%
    select(scenario, method, pearson_mean),
  read.csv(file.path(EXP, "eval/cycif/cycif_unist_summary.csv")))
mk_bar(cy, "scenario", "pearson_mean",
       "CyCIF CRC1 (senario1/2; UniST=fast mode)",
       "fig_cycif_pearson")

## cell typing bars (deepstarmap + wemerfish ARI)
dsc <- read.csv(file.path(EXP, "eval/deepstarmap/celltyping/celltyping_metrics.csv")) %>%
  filter(metric=="ari", scenario %in% c("senario1","senario2")) %>%
  mutate(method=ifelse(method=="UniST","UniST-inr",method))
mk_bar(dsc, "scenario", "value", "Deep-STARmap cell typing (ARI)",
       "fig_deepstarmap_celltyping_ari", ylabel="ARI")

wmc <- read.csv(file.path(EXP, "eval/wemerfish/celltyping/celltyping_metrics.csv")) %>%
  filter(metric=="ari", scenario %in% c("senario1","senario2")) %>%
  mutate(method=ifelse(method=="UniST","UniST-inr",method))
mk_bar(wmc, "scenario", "value", "weMERFISH cell typing (ARI)",
       "fig_wemerfish_celltyping_ari", ylabel="ARI")

wmn <- read.csv(file.path(EXP, "eval/wemerfish/nichetyping/nichetyping_metrics.csv")) %>%
  filter(metric=="ari", scenario %in% c("senario1","senario2")) %>%
  mutate(method=ifelse(method=="UniST","UniST-inr",method))
mk_bar(wmn, "scenario", "value", "weMERFISH niche typing (ARI)",
       "fig_wemerfish_nichetyping_ari", ylabel="ARI")

cat("all figures done\n")
