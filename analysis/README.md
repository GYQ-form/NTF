# NTF Analysis Scripts

The `analysis/` directory contains the final NTF-side scripts used to reproduce the paper's experiments and revision analyses. Unlike the general-purpose tools in `scripts/`, these pipelines are organized by dataset and include experiment-specific preprocessing, evaluation, summarization, and figure generation. Each directory's README documents the required inputs and reproduction commands.

| Directory | Reproduction contents |
|---|---|
| [`simulation/`](simulation/) | Simulation generation; scale, sparse-section, mixed-resolution, gene-count, SVG-conservation, UniST-comparison, and model-ablation analyses. |
| [`STARmap_mouse_VC/`](STARmap_mouse_VC/) | Reconstruction and cell-typing evaluation, model-component ablation, in-silico query/occupancy-filter validation, and spatial scale-bar checks. |
| [`deep_starmap_brain/`](deep_starmap_brain/) | Five-scenario Deep-STARmap benchmark, unified reconstruction/SVG/cell-typing metrics, efficiency summaries, and figures. |
| [`wemerfish_zebrafish_embryo/`](wemerfish_zebrafish_embryo/) | Data preparation, cell/niche-typing evaluation, composition summaries, figures, and validation of the complete notebook retraining workflow. |
| [`cycif_crc1_3d/`](cycif_crc1_3d/) | CRC1 preprocessing and annotation, MAPS alignment, five-scenario NTF reconstruction, cell/niche and SVG evaluation, baseline-feasibility analysis, and figures. |
| [`mouse_embryo/`](mouse_embryo/) | E16.5 cell-type distribution, per-section frequency, 3D morphology comparisons, and summary tables. |

Large datasets and generated benchmark outputs are not committed to this directory; follow the data links and paths in the per-dataset READMEs. For a guided introduction to NTF, use the [tutorials](../README.md#tutorials).
