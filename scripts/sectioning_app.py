import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pickle
import torch
import pandas as pd
import anndata
from scipy.spatial import cKDTree
from scipy.stats import mode
import tempfile
import os

from NTF.sample import sample_points

# --- 1. 加载资产 ---
ASSETS_DIR = '/home/gongyuqiao/ur_annotation/NTF/mytrain/data/sectioning/mouse_embryo'
with open(f'{ASSETS_DIR}/metadata.pkl', 'rb') as f:
    meta = pickle.load(f)

vertices = meta['vertices']
faces = meta['faces']
raw_coords = meta['raw_coords']
avg_dist = meta['avg_dist']
spatial_scaling = meta['spatial_scaling']
gene_names = meta['gene_names']
obs_categories = meta.get('obs_categories', {})  # 字典 {key: {'codes': [], 'categories': []}}
cat_keys = list(obs_categories.keys())

bbox_min = meta['raw_coords_min']
bbox_max = meta['raw_coords_max']
center = (bbox_min + bbox_max) / 2

# 构建 KD 树（既用于边界判定，也用于 KNN 投票）
kdtree = cKDTree(raw_coords)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = f'{ASSETS_DIR}/trained_inr.pt'
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

# --- 2. 辅助函数 ---
def get_plane_vectors(normal):
    normal = normal / np.linalg.norm(normal)
    not_v = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(normal, not_v)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v

def sample_points_on_plane(point, normal, n_spots, bbox_min, bbox_max):
    u, v = get_plane_vectors(normal)
    
    radius = np.linalg.norm(bbox_max - bbox_min)
    # 生成过量的点，因为后续会基于表达量进行大量过滤
    grid_res = int(np.sqrt(n_spots * 10))
    lin = np.linspace(-radius, radius, grid_res)
    U, V = np.meshgrid(lin, lin)
    
    candidates_3d = point + np.outer(U.flatten(), u) + np.outer(V.flatten(), v)
    
    # 这里使用一个非常宽泛的物理阈值(20倍)仅仅是为了剔除距离组织极其遥远的无效空间，节省GPU显存
    distances, _ = kdtree.query(candidates_3d)
    mask_generous = distances < (avg_dist * 20.0)
    
    valid_points_3d = candidates_3d[mask_generous]
    valid_uv = np.column_stack((U.flatten()[mask_generous], V.flatten()[mask_generous]))
    
    # 随机保留最多 n_spots * 3 个候选点送入模型预测
    max_candidates = n_spots * 3
    if len(valid_points_3d) > max_candidates:
        idx = np.random.choice(len(valid_points_3d), max_candidates, replace=False)
        return valid_points_3d[idx], valid_uv[idx]
    
    return valid_points_3d, valid_uv


# --- 3. Dash UI 构建 ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H2("NTF In-silico Sectioning", className="mt-2 mb-3 text-center"),
    
    dbc.Row([
        # ================= 左侧面板 =================
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("3D Tissue Model & Plane Preview", className="font-weight-bold"),
                dbc.CardBody(dcc.Graph(id='plot-3d', style={'height': '70vh'}), className="p-2")
            ], className="mb-3 shadow-sm"),
            
            dbc.Card([
                dbc.CardHeader("1. Define Sectioning Plane", className="font-weight-bold bg-light py-2"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Normal Vector (nx, ny, nz)", className="small mb-1"),
                            dbc.InputGroup([
                                dbc.Input(id='nx', type='number', value=0, step=0.1, size="sm"),
                                dbc.Input(id='ny', type='number', value=0, step=0.1, size="sm"),
                                dbc.Input(id='nz', type='number', value=1, step=0.1, size="sm"),
                            ])
                        ], width=5),
                        dbc.Col([
                            html.Label("Point on Plane (x, y, z)", className="small mb-1"),
                            dbc.InputGroup([
                                dbc.Input(id='px', type='number', value=center[0], size="sm"),
                                dbc.Input(id='py', type='number', value=center[1], size="sm"),
                                dbc.Input(id='pz', type='number', value=center[2], size="sm"),
                            ])
                        ], width=5),
                        dbc.Col([
                            html.Label("\u00A0", className="small mb-1 d-block"), 
                            dbc.Button("Preview", id="btn-preview", color="info", size="sm", className="w-100")
                        ], width=2)
                    ])
                ], className="py-2")
            ], className="mb-2 shadow-sm"),
        ], width=6),
        
        # ================= 右侧面板 =================
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    dbc.Row([
                        dbc.Col("2D Prediction Viewer", className="font-weight-bold mt-1", width=5),
                        dbc.Col(
                            dcc.RadioItems(
                                id='view-mode',
                                options=[
                                    {'label': ' Gene Expression ', 'value': 'gene'},
                                    {'label': ' Categorical ', 'value': 'cat'}
                                ],
                                value='gene',
                                inline=True,
                                inputClassName="me-1",
                                labelClassName="me-3 small font-weight-bold"
                            ), width=7, className="text-end"
                        )
                    ]), className="py-2"
                ),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading-2d-plot", type="circle",
                        children=dcc.Graph(id='plot-2d', style={'height': '70vh'})
                    )
                ], className="p-2")
            ], className="mb-3 shadow-sm"),
            
            dbc.Card([
                dbc.CardHeader("2. Generate Slice & Inference", className="font-weight-bold bg-light py-2"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Spots Target", className="small mb-1"),
                            dbc.Input(id='n_spots', type='number', value=5000, step=100, size="sm"),
                        ], width=2),
                        dbc.Col([
                            html.Label("\u00A0", className="small mb-1 d-block"),
                            dbc.Button("Predict", id="btn-generate", color="primary", size="sm", className="w-100"),
                        ], width=3),
                        
                        # 动态显示基因或分类下拉框
                        dbc.Col([
                            html.Label("Select Target", className="small mb-1"),
                            html.Div(
                                dcc.Dropdown(id='gene-dropdown', options=[{'label': g, 'value': g} for g in gene_names], value=gene_names[0] if len(gene_names)>0 else None, optionHeight=30, style={'height': '30px', 'minHeight': '30px', 'lineHeight': '1.5'}),
                                id="div-gene-dropdown"
                            ),
                            html.Div(
                                dcc.Dropdown(id='cat-dropdown', options=[{'label': c, 'value': c} for c in cat_keys], value=cat_keys[0] if len(cat_keys)>0 else None, optionHeight=30, style={'height': '30px', 'minHeight': '30px', 'lineHeight': '1.5'}),
                                id="div-cat-dropdown", style={'display': 'none'}
                            )
                        ], width=4),
                        
                        dbc.Col([
                            html.Label("\u00A0", className="small mb-1 d-block"),
                            dbc.Button("Download .h5ad", id="btn-download", color="success", size="sm", outline=True, className="w-100"),
                            dcc.Download(id="download-h5ad")
                        ], width=3)
                    ]),
                    dcc.Loading(
                        id="loading-inference", type="dot",
                        children=html.Div(id="inference-status", className="text-center text-success small mt-2", style={"minHeight": "18px"})
                    ),
                ], className="py-2")
            ], className="mb-2 shadow-sm"),
        ], width=6)
    ]),
    
    dcc.Store(id='store-prediction-data')
    
], fluid=True, className="px-4 py-2")


# --- 4. 回调函数 ---

# 切换下拉菜单的显示/隐藏
@app.callback(
    [Output('div-gene-dropdown', 'style'),
     Output('div-cat-dropdown', 'style')],
    [Input('view-mode', 'value')]
)
def toggle_dropdowns(view_mode):
    if view_mode == 'gene':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}


@app.callback(
    Output('plot-3d', 'figure'),
    [Input('btn-preview', 'n_clicks')],
    [State('nx', 'value'), State('ny', 'value'), State('nz', 'value'),
     State('px', 'value'), State('py', 'value'), State('pz', 'value')]
)
def update_3d_preview(n_clicks, nx, ny, nz, px, py, pz):
    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            opacity=0.3, color='lightpink', name='Tissue Surface'
        )
    ])
    normal = np.array([nx, ny, nz])
    if np.linalg.norm(normal) > 0:
        point = np.array([px, py, pz])
        u, v = get_plane_vectors(normal)
        size = np.linalg.norm(bbox_max - bbox_min) * 0.8
        corners = np.array([
            point + size*u + size*v, point - size*u + size*v,
            point - size*u - size*v, point + size*u - size*v
        ])
        fig.add_trace(go.Mesh3d(
            x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
            i=[0, 0], j=[1, 2], k=[2, 3],
            opacity=0.5, color='cyan', name='Sectioning Plane'
        ))
    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
    return fig


@app.callback(
    [Output('store-prediction-data', 'data'), Output('inference-status', 'children')],
    [Input('btn-generate', 'n_clicks')],
    [State('nx', 'value'), State('ny', 'value'), State('nz', 'value'),
     State('px', 'value'), State('py', 'value'), State('pz', 'value'),
     State('n_spots', 'value')],
    prevent_initial_call=True
)
def generate_slice_and_predict(n_clicks, nx, ny, nz, px, py, pz, n_spots):
    normal = np.array([nx, ny, nz])
    point = np.array([px, py, pz])
    
    # 宽泛采样
    sampled_3d, sampled_uv = sample_points_on_plane(point, normal, n_spots, bbox_min, bbox_max)
    if len(sampled_3d) == 0:
        return None, "⚠️ No points found inside the bounding box for this plane."
        
    scaled_coords = sampled_3d * spatial_scaling
    coords_tensor = torch.from_numpy(scaled_coords).float().to(device)
    
    # 1. 模型预测表达
    with torch.no_grad():
        prediction_results = sample_points(model, coords_tensor)
        predicted_expression = prediction_results["expression"].cpu().numpy()
    
    # ================= 优化2：基于表达量自适应剔除背景 =================
    # 计算每个点的总基因表达量
    total_expr_per_spot = predicted_expression.sum(axis=1)
    
    # 设定阈值：保留总表达量大于最高点 5% 的点
    threshold = np.max(total_expr_per_spot) * 0.05
    keep_mask = total_expr_per_spot > threshold
    
    sampled_3d = sampled_3d[keep_mask]
    sampled_uv = sampled_uv[keep_mask]
    predicted_expression = predicted_expression[keep_mask]
    
    if len(sampled_3d) == 0:
        return None, "⚠️ All sampled points had near-zero expression (outside tissue). Try moving the plane."
        
    # 如果过滤后剩下的点仍然过多，随机下采样到目标数量
    if len(sampled_3d) > n_spots:
        idx = np.random.choice(len(sampled_3d), n_spots, replace=False)
        sampled_3d = sampled_3d[idx]
        sampled_uv = sampled_uv[idx]
        predicted_expression = predicted_expression[idx]
    # ===================================================================

    # # ================= 优化1：基因维度的标准化(Min-Max) =================
    # # 按列(基因)求最大最小值
    # min_vals = predicted_expression.min(axis=0)
    # max_vals = predicted_expression.max(axis=0)
    # range_vals = max_vals - min_vals
    # range_vals[range_vals == 0] = 1.0  # 防止除以 0
    
    # # 标准化：将每个基因的表达映射到 0 ~ 1 之间
    # normalized_expression = (predicted_expression - min_vals) / range_vals
    # # ===================================================================

    # ================= 优化1：0.1-1.0 非零映射标准化 =================
    # 设定极小值阈值，判定什么是"零"
    eps = 1e-6
    is_nonzero = predicted_expression > eps
    
    # 找到非零的最小值
    masked_expr = np.where(is_nonzero, predicted_expression, np.inf)
    min_vals_nonzero = masked_expr.min(axis=0)
    min_vals_nonzero[np.isinf(min_vals_nonzero)] = 0.0  # 处理全零基因
    
    # 找到最大值并计算极差
    max_vals = predicted_expression.max(axis=0)
    range_vals = max_vals - min_vals_nonzero
    range_vals[range_vals <= 0] = 1.0  # 防止除以 0
    
    # 初始化一个全 0 矩阵
    normalized_expression = np.zeros_like(predicted_expression)
    
    # 对所有点进行 0.1 ~ 1.0 的映射计算
    # 公式: 0.1 + 0.9 * (x - min) / (max - min)
    scaled_expr = 0.1 + 0.9 * ((predicted_expression - min_vals_nonzero) / range_vals)
    
    # 仅将原本非零的点替换为映射后的值，原本为0的点依���保持为0
    normalized_expression[is_nonzero] = scaled_expr[is_nonzero]
    
    # 兜底截断，确保数值绝对安全
    normalized_expression = np.clip(normalized_expression, 0.0, 1.0)
    # ===================================================================

    # 2. KNN 投票预测定性变量 (仅对过滤后保留的实体组织点计算)
    pred_obs = {}
    if obs_categories:
        k_neighbors = min(5, len(raw_coords))
        _, indices = kdtree.query(sampled_3d, k=k_neighbors)
        if k_neighbors == 1: indices = indices.reshape(-1, 1)
        
        for key, info in obs_categories.items():
            codes = info['codes']
            categories = np.array(info['categories'])
            
            neighbor_codes = codes[indices]
            try:
                voted_modes, _ = mode(neighbor_codes, axis=1, keepdims=False)
            except TypeError:
                voted_modes, _ = mode(neighbor_codes, axis=1)
                
            pred_obs[key] = categories[voted_modes.flatten()].tolist()
    
    data_dict = {
        'x_2d': sampled_uv[:, 0].tolist(), 'y_2d': sampled_uv[:, 1].tolist(),
        'x_3d': sampled_3d[:, 0].tolist(), 'y_3d': sampled_3d[:, 1].tolist(), 'z_3d': sampled_3d[:, 2].tolist(),
        'expression': normalized_expression.tolist(), # 发送给前端的是标准化后的平滑数据
        'pred_obs': pred_obs  
    }
    return data_dict, f"✅ Inference complete! Retained {len(sampled_3d)} confident tissue spots."


@app.callback(
    Output('plot-2d', 'figure'),
    [Input('view-mode', 'value'),
     Input('gene-dropdown', 'value'),
     Input('cat-dropdown', 'value'),
     Input('store-prediction-data', 'data')],
    prevent_initial_call=True
)
def update_2d_plot(view_mode, selected_gene, selected_cat, data):
    if not data:
        return go.Figure()
        
    if view_mode == 'gene' and selected_gene:
        gene_idx = gene_names.index(selected_gene)
        expr = [row[gene_idx] for row in data['expression']]
        fig = px.scatter(
            x=data['x_2d'], y=data['y_2d'], color=expr,
            color_continuous_scale='Viridis',
            title=f"Relative Expression: {selected_gene}"
        )
    elif view_mode == 'cat' and selected_cat and selected_cat in data.get('pred_obs', {}):
        cat_labels = data['pred_obs'][selected_cat]
        # 使用大量离散颜色的拼接，以防类别太多
        rich_palette = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
        
        fig = px.scatter(
            x=data['x_2d'], y=data['y_2d'], color=cat_labels,
            color_discrete_sequence=rich_palette,
            title=f"Annotation: {selected_cat}"
        )
    else:
        return go.Figure()
        
    fig.update_layout(
        yaxis=dict(scaleanchor="x", scaleratio=1), 
        margin=dict(l=0, r=0, b=0, t=40),
        legend_title_text=''
    )
    # 把散点稍微调大一点，方便看清颜色
    fig.update_traces(marker=dict(size=6))
    return fig


@app.callback(
    Output("download-h5ad", "data"),
    Input("btn-download", "n_clicks"),
    State('store-prediction-data', 'data'),
    prevent_initial_call=True
)
def download_anndata(n_clicks, data):
    if not data:
        return dash.no_update
        
    # 下载的也是标准化后的表达矩阵，方便下游统一作图
    expr_matrix = np.array(data['expression'])
    coords_3d = np.column_stack((data['x_3d'], data['y_3d'], data['z_3d']))
    coords_2d = np.column_stack((data['x_2d'], data['y_2d']))
    
    obs_df = pd.DataFrame(index=[f"spot_{i}" for i in range(len(coords_3d))])
    
    # 填入预测的分类变量
    if 'pred_obs' in data:
        for key, labels in data['pred_obs'].items():
            obs_df[key] = labels
            obs_df[key] = obs_df[key].astype('category')
            
    adata_slice = anndata.AnnData(X=expr_matrix, obs=obs_df)
    adata_slice.var_names = gene_names
    adata_slice.obsm['spatial_3d'] = coords_3d
    adata_slice.obsm['spatial_2d'] = coords_2d
    
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        adata_slice.write_h5ad(tmp_path)
        with open(tmp_path, "rb") as f:
            data_bytes = f.read()
        return dcc.send_bytes(data_bytes, "virtual_slice.h5ad")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == '__main__':
    app.run(debug=True, port=8201)