import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import anndata
import scanpy as sc
import argparse
import scipy.sparse as sp
import matplotlib.pyplot as plt

# --- 参数解析 ---
parser = argparse.ArgumentParser(description="单数据三维可视化")
parser.add_argument('--obs-cols', nargs='+', default=['cell_type', 'leiden', 'cluster', 'domain', 'slice_id','slice','annotation','slice_ID','data_type'], help="obs列名")
parser.add_argument('--port', type=int, default=8060, help="端口")
parser.add_argument('--base-url', type=str, default='/', help="URL 路径前缀，例如 /app1/")
parser.add_argument('--title', type=str, default='三维空间转录组可视化', help="展示标题")
parser.add_argument('-i',"--input_path", type=str, required=True, help="h5ad路径")
args = parser.parse_args()

# --- 数据加载逻辑 ---
print(f"正在加载数据: {args.input_path}")
adata_orig = sc.read_h5ad(args.input_path)

# 下采样
SAMPLING_THRESHOLD = 100000
if adata_orig.n_obs > SAMPLING_THRESHOLD:
    adata = sc.pp.subsample(adata_orig, n_obs=SAMPLING_THRESHOLD, copy=True)
    print(f"数据量大于 {SAMPLING_THRESHOLD}，已自动下采样。")
else:
    adata = adata_orig

df = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y', 'z'], index=adata.obs.index)
valid_obs_cols = [col for col in args.obs_cols if col in adata.obs.columns]
if not valid_obs_cols and 'cell_type' in adata.obs: valid_obs_cols = ['cell_type']
if valid_obs_cols: df = df.join(adata.obs[valid_obs_cols])

adata.var_names = adata.var.gene_symbol if 'gene_symbol' in adata.var.columns else adata.var_names
gene_list = adata.var_names.tolist()

# --- Dash App ---
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'],
                url_base_pathname=args.base_url)

SIDEBAR_STYLE = {
    "position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "20rem",
    "padding": "2rem 1rem", "background-color": "#f8f9fa", "overflow-y": "auto",
    "transition": "margin-left .5s", "z-index": 1000, "box-shadow": "2px 0 5px rgba(0,0,0,0.1)"
}
CONTENT_STYLE = {
    "margin-left": "22rem", "margin-right": "2rem", "padding": "2rem 1rem", "transition": "margin-left .5s"
}

app.layout = html.Div([
    dcc.Store(id='sidebar-state', data=True),

    # --- 左侧控制栏 ---
    html.Div(id="sidebar", style=SIDEBAR_STYLE, children=[
        html.H4("控制面板", className="display-4"),
        html.Hr(),
        html.Button('收起 / 展开', id='toggle-sidebar', n_clicks=0, className="button-primary", style={'width': '100%', 'margin-bottom': '10px'}),
        
        html.Label("显示模式:"),
        dcc.Dropdown(
            id='view-selector',
            options=[{'label': '分类变量', 'value': 'categorical'}, {'label': '基因表达 (支持多选)', 'value': 'gene_expression'}],
            value='categorical', clearable=False
        ),
        html.Br(),
        
        html.Label("选择目标 (可多选):"),
        dcc.Dropdown(id='item-selector', clearable=False),
        
        # 基因表达专用的过滤控制面板
        html.Div(id='gene-specific-controls', children=[
            html.Br(),
            html.Div(style={'padding': '10px', 'background-color': '#e9ecef', 'border-radius': '5px'}, children=[
                html.Strong("表达量过滤:"),
                
                # 过滤参考基因下拉框
                html.Label("过滤参考基因 (可选):", style={'margin-top': '10px', 'font-size': '13px'}),
                html.Div("若选择，则下方阈值筛选作用于该基因", style={'font-size': '11px', 'color': '#666', 'margin-bottom': '5px'}),
                dcc.Dropdown(
                    id='filter-ref-gene', 
                    options=[{'label': g, 'value': g} for g in gene_list], 
                    value=None, 
                    placeholder="默认: 按自身表达量过滤", 
                    clearable=True
                ),
                
                html.Div("提示：多基因同视时建议勾选隐藏0值", style={'font-size': '11px', 'color': '#666', 'margin-top':'8px'}),
                dcc.Checklist(
                    id='filter-zero',
                    options=[{'label': ' 隐藏表达量为 0 的点', 'value': 'hide_zero'}],
                    value=[], 
                    style={'margin-top': '2px', 'margin-bottom': '8px'}
                ),
                html.Label("最低表达阈值 (绝对值):"),
                dcc.Input(id='expr-threshold', type='number', value=0.0, step=0.1, style={'width': '100%'}),
                
                # --- 新增：适应性透明度 ---
                html.Hr(style={'margin-top': '10px', 'margin-bottom': '10px'}),
                html.Strong("高级视觉渲染:"),
                dcc.Checklist(
                    id='adaptive-opacity',
                    options=[{'label': ' 适应性透明度 (表达量越低越透明)', 'value': 'adaptive'}],
                    value=[], 
                    style={'margin-top': '5px', 'margin-bottom': '5px'}
                )
            ])
        ], style={'display': 'none'}),
        
        html.Br(),
        html.Label("点大小 (Size):"),
        dcc.Slider(id='size-slider', min=1, max=10, step=0.5, value=3, marks={1:'1', 5:'5', 10:'10'}),
        html.Br(),

        html.Label("全局最高透明度 (Opacity上限):"),
        dcc.Slider(id='opacity-slider', min=0.1, max=1, step=0.1, value=1.0, marks={0.1:'0.1', 1:'1'}),
        html.Hr(),

        html.Label("视觉选项:"),
        dcc.Checklist(
            id='visual-options',
            options=[
                {'label': ' 隐藏坐标轴背景 (沉浸模式)', 'value': 'hide_bg'},
            ],
            value=[]
        )
    ]),

    # --- 右侧内容区 ---
    html.Div(id="page-content", style=CONTENT_STYLE, children=[
        html.H2(args.title, style={'text-align': 'center'}),
        dcc.Loading(
            type="circle",
            children=dcc.Graph(id='spatial-3d-plot', style={'height': '85vh'})
        )
    ])
])

# --- Callbacks ---

@app.callback(
    [Output("sidebar", "style"), Output("page-content", "style")],
    [Input("toggle-sidebar", "n_clicks")],
    [State("sidebar", "style"), State("page-content", "style")]
)
def toggle_sidebar(n_clicks, sidebar_style, content_style):
    if n_clicks % 2 == 1: 
        sidebar_style['margin-left'] = "-20rem"
        content_style['margin-left'] = "2rem"
    else: 
        sidebar_style['margin-left'] = "0"
        content_style['margin-left'] = "22rem"
    return sidebar_style, content_style

@app.callback(
    [Output('item-selector', 'options'),
     Output('item-selector', 'value'),
     Output('item-selector', 'multi'),  
     Output('gene-specific-controls', 'style')],
    [Input('view-selector', 'value')]
)
def update_selector_and_controls(view_mode):
    if view_mode == 'gene_expression':
        opts = [{'label': g, 'value': g} for g in gene_list]
        val = [gene_list[0]] if gene_list else []
        return opts, val, True, {'display': 'block'}  
    else:
        opts = [{'label': c, 'value': c} for c in valid_obs_cols]
        val = valid_obs_cols[0] if valid_obs_cols else None
        return opts, val, False, {'display': 'none'}   

# 核心绘图逻辑
@app.callback(
    Output('spatial-3d-plot', 'figure'),
    [Input('view-selector', 'value'),
     Input('item-selector', 'value'),
     Input('size-slider', 'value'),
     Input('opacity-slider', 'value'),
     Input('visual-options', 'value'),
     Input('filter-ref-gene', 'value'),    
     Input('filter-zero', 'value'),        
     Input('expr-threshold', 'value'),
     Input('adaptive-opacity', 'value')],
    [State('spatial-3d-plot', 'relayoutData')]
)
def update_graph(view_mode, selected_item, size, opacity, visual_opts, filter_ref_gene, filter_zero, expr_threshold, adaptive_opacity, relayout_data):
    if not selected_item: return go.Figure()

    camera = relayout_data['scene.camera'] if relayout_data and 'scene.camera' in relayout_data else None
    hide_bg = 'hide_bg' in visual_opts
    axis_template = dict(
        showgrid=not hide_bg, showbackground=not hide_bg, 
        showticklabels=not hide_bg, title='' if hide_bg else None, visible=not hide_bg
    )
    scene_settings = dict(aspectmode='data', xaxis=axis_template, yaxis=axis_template, zaxis=axis_template, camera=camera)
    bg_color = "rgba(0,0,0,0)" if hide_bg else "white"

    if view_mode == 'gene_expression':
        genes = selected_item if isinstance(selected_item, list) else [selected_item]
        if len(genes) > 5: genes = genes[:5]
            
        color_scales = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges']
        traces = []
        
        # 提前提取参考基因的表达量
        ref_expr = None
        if filter_ref_gene and filter_ref_gene in adata.var_names:
            ref_data = adata[:, filter_ref_gene].X
            ref_expr = ref_data.toarray().flatten() if sp.issparse(ref_data) else np.asarray(ref_data).flatten()
        
        for i, gene in enumerate(genes):
            # 目标基因表达量
            expr_data = adata[:, gene].X
            expr = expr_data.toarray().flatten() if sp.issparse(expr_data) else np.asarray(expr_data).flatten()
            
            mask_target_expr = ref_expr if ref_expr is not None else expr
            
            mask = np.ones_like(expr, dtype=bool)
            if 'hide_zero' in filter_zero: mask &= (mask_target_expr > 0)
            if expr_threshold is not None and expr_threshold > 0: mask &= (mask_target_expr >= expr_threshold)
                
            if mask.sum() == 0:
                continue
                
            df_filtered = df[mask]
            expr_filtered = expr[mask]

            cscale = 'Viridis' if len(genes) == 1 else color_scales[i % len(color_scales)]
            cbar_x = 1.0 if len(genes) == 1 else (1.02 + (i * 0.08))

            # ================= 适应性透明度修复逻辑 =================
            if adaptive_opacity and 'adaptive' in adaptive_opacity:
                e_min = expr_filtered.min()
                e_max = expr_filtered.max()
                
                # 计算 0~1 的标准化表达量
                if e_max > e_min:
                    norm_expr = (expr_filtered - e_min) / (e_max - e_min)
                else:
                    norm_expr = np.ones_like(expr_filtered)
                
                # 动态 Alpha 通道
                dynamic_alpha = norm_expr * opacity
                
                # 使用 matplotlib 获取色谱的 RGB 矩阵
                cmap_name = cscale.lower() if cscale == 'Viridis' else cscale
                cmap = plt.get_cmap(cmap_name)
                rgba_matrix = cmap(norm_expr)
                
                # 将算好的动态 Alpha 强行注入第四通道
                rgba_matrix[:, 3] = dynamic_alpha
                
                # 转换为 Plotly 认识的 rgba(r,g,b,a) 字符串数组
                color_array = [f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.3f})' for r,g,b,a in rgba_matrix]
                
                # 1. 真实的三维散点 (使用 rgba 数组，它自带颜色和透明度)
                traces.append(go.Scatter3d(
                    x=df_filtered['x'], y=df_filtered['y'], z=df_filtered['z'],
                    mode='markers', name=gene, 
                    marker=dict(size=size, color=color_array),
                    hovertext=[f'{gene}: {v:.3f}' for v in expr_filtered], hoverinfo='text'
                ))
                
                # 2. 隐藏的 Dummy 轨迹 (唯一作用是为了在侧边挂载 Colorbar)
                traces.append(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers', showlegend=False, hoverinfo='none',
                    marker=dict(
                        size=0, 
                        color=[expr.min(), expr.max()], # 提供真实的全局极值以便刻度正确
                        colorscale=cscale,
                        colorbar=dict(title=gene, x=cbar_x, thickness=12, len=0.7),
                        showscale=True
                    )
                ))
                
            else:
                # ================= 原有的统��透明度逻辑 =================
                traces.append(go.Scatter3d(
                    x=df_filtered['x'], y=df_filtered['y'], z=df_filtered['z'],
                    mode='markers',
                    name=gene, 
                    marker=dict(
                        size=size, opacity=opacity,
                        color=expr_filtered, colorscale=cscale,
                        colorbar=dict(title=gene, x=cbar_x, thickness=12, len=0.7),
                        showscale=True
                    ),
                    hovertext=[f'{gene}: {v:.3f}' for v in expr_filtered], hoverinfo='text'
                ))

        if not traces:
            return go.Figure(layout={'title': "在当前过滤条件下无数据点"})
            
        fig = go.Figure(data=traces)
        
        # 动态标题
        title_suffix = []
        if 'hide_zero' in filter_zero: title_suffix.append("隐藏0值")
        if expr_threshold is not None and expr_threshold > 0: title_suffix.append(f"阈值 ≥ {expr_threshold}")
        if adaptive_opacity and 'adaptive' in adaptive_opacity: title_suffix.append("自适应透明")
        
        filter_target_name = filter_ref_gene if filter_ref_gene else "自身"
        genes_str = ", ".join(genes)
        
        if title_suffix:
            title = f"基因: {genes_str} (按 {filter_target_name} 过滤: {', '.join(title_suffix)})"
        else:
            title = f"基因: {genes_str}"
            
        if len(selected_item) > 5: title += " | 最多显示前5个"
    
    else:
        # 分类变量部分保持不变...
        cat_item = selected_item[0] if isinstance(selected_item, list) else selected_item
        plot_data = df.copy()
        plot_data[cat_item] = plot_data[cat_item].astype(str)
        rich_palette = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
        
        fig = px.scatter_3d(plot_data, x='x', y='y', z='z', color=cat_item,
                            color_discrete_sequence=rich_palette,
                            category_orders={cat_item: sorted(plot_data[cat_item].unique())})
        fig.update_traces(marker=dict(size=size, opacity=opacity))
        title = f"分类: {cat_item}"

    fig.update_layout(
        title=title, scene=scene_settings, 
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor=bg_color,
        showlegend=True,
        legend=dict(font=dict(size=14), itemsizing='constant', yanchor="top", y=0.9, xanchor="left", x=0.05)
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=args.port, host='0.0.0.0')