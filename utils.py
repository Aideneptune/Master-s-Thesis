import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import config

def format_time(seconds):
    """Másodpercek formázása olvasható szöveggé."""
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes = int(seconds // 60)
    sec = seconds % 60
    return f"{minutes}m {sec:.0f}s"

def create_features(df):
    """Származtatott jellemzők generálása."""
    df['Log_Time'] = np.log(df['Time'])
    df['Time_Squared'] = df['Time'] ** 2
    
    # --- Hertz-féle maximális feszültség számítása ---
    E_star = config.E_MODULUS / (2.0 * (1.0 - config.POISSON_RATIO**2))
    a = np.cbrt((3.0 * df['Load'] * config.BALL_RADIUS) / (4.0 * E_star))
    df['Hertz_Stress_MPa'] = (3.0 * df['Load']) / (2.0 * np.pi * a**2)
    
    return df

def filter_outliers_grouped(df, group_col, cols, low_q=0.01, high_q=0.99):
    """Percentilis szűrés (Outlierek eltávolítása) csoportonként."""
    mask = pd.Series(True, index=df.index)
    for col in cols:
        bounds = df.groupby(group_col)[col].quantile([low_q, high_q]).unstack()
        lowers = df[group_col].map(bounds[low_q])
        highers = df[group_col].map(bounds[high_q])
        mask &= (df[col] >= lowers) & (df[col] <= highers)
    return df[mask]

def plot_pareto_front(results_dir, predictions, temperatures, title="Pareto front"):
    """Pareto front vizualizáció és térdkapu pont keresés."""
    cof = predictions[:, 0]
    fai = predictions[:, 1]
    costs = np.column_stack((cof, fai))
    
    pareto_indices = []
    for i in range(costs.shape[0]):
        dominated = False
        for j in range(costs.shape[0]):
            if i == j: continue
            if (costs[j, 0] <= costs[i, 0] and costs[j, 1] <= costs[i, 1]) and \
               (costs[j, 0] < costs[i, 0] or costs[j, 1] < costs[i, 1]):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)
            
    pareto_cof = cof[pareto_indices]
    pareto_fai = fai[pareto_indices]
    
    sorted_indices = np.argsort(pareto_cof)
    pareto_cof = pareto_cof[sorted_indices]
    pareto_fai = pareto_fai[sorted_indices]

    if len(pareto_cof) > 2:
        p_cof_norm = (pareto_cof - pareto_cof.min()) / (pareto_cof.max() - pareto_cof.min() + 1e-9)
        p_fai_norm = (pareto_fai - pareto_fai.min()) / (pareto_fai.max() - pareto_fai.min() + 1e-9)
        p1 = np.array([p_cof_norm[0], p_fai_norm[0]])
        p2 = np.array([p_cof_norm[-1], p_fai_norm[-1]])
        vec_line = p2 - p1
        
        distances = []
        for i in range(len(pareto_cof)):
            p0 = np.array([p_cof_norm[i], p_fai_norm[i]])
            vec_point = p0 - p1
            dist = np.abs(vec_line[0] * vec_point[1] - vec_line[1] * vec_point[0]) / np.linalg.norm(vec_line)
            distances.append(dist)
            
        knee_idx = np.argmax(distances)
        knee_cof = pareto_cof[knee_idx]
        knee_fai = pareto_fai[knee_idx]
    else:
        knee_cof, knee_fai = pareto_cof[0], pareto_fai[0]

    plt.figure(figsize=(6.3, 3.15))
    plt.scatter(cof, fai, alpha=0.6, c=temperatures, cmap='plasma', label='Feasible operating points', s=10)
    plt.colorbar(label='Temperature [°C]')
    plt.plot(pareto_cof, pareto_fai, color='purple', marker='o', label='Pareto front (Trade-off optimums)')
    plt.plot(knee_cof, knee_fai, color='orange', marker='o', markersize=10, label='Knee point (Best trade-off)', markeredgecolor='black')
    plt.annotate(f"Knee\n({knee_cof:.3f}, {knee_fai:.3f})", (knee_cof, knee_fai), 
                 xytext=(15, 15), textcoords='offset points', 
                 arrowprops=dict(arrowstyle="->", color='orange'),
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5, alpha=0.7))
    plt.xlabel('Coefficient of friction (COF) [-]')
    plt.ylabel('Friction Absolute Integral [-]')
    ymin, ymax = plt.gca().get_ylim()
    plt.ylim(ymin, ymax + (ymax - ymin) * 0.35)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(results_dir, "Pareto_Optimization.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
    plt.savefig(os.path.join(results_dir, "Pareto_Optimization.svg"), format='svg', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_learning_curve(estimator, X, y, cv=None, n_jobs=-1, train_sizes=np.linspace(0.2, 1.0, 10), results_dir=".", groups=None, num_files=1):
    """Tanulási görbe generálása és mentése."""
    plt.figure(figsize=(6.3, 3.15))
    plt.xlabel("Number of training files/experiments")
    plt.ylabel(r"$R^2$ score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='r2', groups=groups
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    avg_samples_per_file = len(X) / num_files
    train_sizes_files = train_sizes / avg_samples_per_file

    plt.fill_between(train_sizes_files, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="purple")
    plt.fill_between(train_sizes_files, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="orange")
    plt.plot(train_sizes_files, train_scores_mean, 'o-', color="purple", label="Training score")
    plt.plot(train_sizes_files, test_scores_mean, 'o-', color="orange", label="Cross-validation score")
    plt.ylim(max(0.0, np.min(test_scores_mean) - 0.1), 1.25) # Extrában megemelve
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(results_dir, "Learning_Curve.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
    plt.savefig(os.path.join(results_dir, "Learning_Curve.svg"), format='svg', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def generate_html_report(results, xlsx_files, full_df, desc_df, html_path, results_dir, doe_suggestions, optimum_results, shap_text="", timing_stats=None, dynamic_descriptions=None, distribution_summary=None, plotly_3d_html=None):
    """HTML jelentés generálása a megadott PDF struktúra alapján."""
    sorted_results = sorted(results, key=lambda x: x['R2_Test'], reverse=True)
    best_model_res = sorted_results[0]
    
    desc_map = config.IMAGE_DESCRIPTIONS.copy()
    if dynamic_descriptions:
        desc_map.update(dynamic_descriptions)
    
    # CSS és Fejléc
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Results - Summary</title>
    <style>
        body {{ font-family: 'Times New Roman', serif; margin: 0; padding: 20px; background-color: #fff; color: #000; }}
        h1, h2 {{ color: #000; border-bottom: 1px solid #000; padding-bottom: 5px; margin-top: 30px; }}
        h3 {{ color: #000; margin-top: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; font-size: 11pt; }}
        th, td {{ border: 1px solid #000; padding: 8px; text-align: right; }}
        th {{ background-color: #f2f2f2; font-weight: bold; text-align: center; }}
        td:first-child, th:first-child {{ text-align: left; }}
        .img-grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; margin-top: 20px; }}
        .img-card {{ text-align: center; page-break-inside: avoid; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ccc; }}
        .footer {{ margin-top: 40px; text-align: center; font-size: 0.9em; color: #000; }}
        .details-box {{ border: 1px solid #000; padding: 15px; margin-bottom: 15px; text-align: left; page-break-inside: avoid; }}
        ul {{ list-style-type: disc; margin-left: 20px; }}
        li {{ margin-bottom: 5px; }}
        .feat-table {{ width: auto; margin-top: 10px; }}
        .feat-table th, .feat-table td {{ padding: 4px 12px; }}
        .export-btn {{ padding: 10px 20px; font-size: 16px; cursor: pointer; background-color: #4CAF50; color: white; border: none; border-radius: 5px; margin-bottom: 20px; }}
        .export-btn:hover {{ background-color: #45a049; }}
        @media print {{
            .no-print {{ display: none !important; }}
            div {{ overflow: visible !important; max-height: none !important; }}
            table {{ page-break-inside: auto; width: 100%; font-size: 9pt; }}
            tr {{ page-break-inside: avoid; page-break-after: auto; }}
            body {{ padding: 0; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Research results summary</h1>
        <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="details-box" style="background-color: #f9f9f9;">
            <h2>Table of Contents</h2>
            <ul style="list-style-type: none; padding-left: 0;">
                <li><a href="#sec1">1. Settings and information</a></li>
                <li><a href="#sec2">2. Dataset descriptive statistics</a></li>
                <li><a href="#sec3">3. Model results comparison</a></li>
                <li><a href="#sec4">4. Detailed model results</a></li>
                <li><a href="#sec5">5. Generated diagrams</a></li>
                <li><a href="#sec6">6. Evaluation and Optimum Comparison</a></li>
                <li><a href="#sec7">7. DoE Suggestions (New measurements)</a></li>
            </ul>
        </div>
        
        <div class="footer no-print" style="margin-top: 20px;">
            <button class="export-btn" onclick="window.print()">Export / Print to PDF</button>
        </div>
        
        <h2 id="sec1">1. Settings and information</h2>
        <ul>
            <li><strong>Number of processed files:</strong> {len(xlsx_files)}</li>
            <li><strong>Dataset size:</strong> {len(full_df)} rows</li>
            <li><strong>Train/Test split:</strong> 80%/20%</li>
            <li><strong>Cross-validation (CV):</strong> GroupKFold (10-fold)</li>
            <li><strong>Hyperparameter optimization:</strong> Optuna Bayesian Optimization (TPE, 50 trials, 10-fold CV)</li>"""
            
    if timing_stats:
        html_content += f"""
            <li><strong>Total script execution time:</strong> {timing_stats.get('total', 'N/A')}</li>
            <li><strong>Data loading time:</strong> {timing_stats.get('loading', 'N/A')}</li>
            <li><strong>SHAP analysis time:</strong> {timing_stats.get('shap', 'N/A')}</li>
            <li><strong>DoE calculation time:</strong> {timing_stats.get('doe', 'N/A')}</li>"""
            
    html_content += f"""
        </ul>
        
        <div class="details-box" style="background-color: #f4f7f6; border-left: 4px solid #4CAF50; margin-top: 20px;">
            <h3 style="margin-top: 0;">Hertzian Maximum Contact Stress Formula</h3>
            <p>The maximum contact stress (<strong>P<sub>max</sub></strong>) for a ball-on-flat contact is calculated as:</p>
            <p style="text-align: center; font-size: 1.2em; margin: 15px 0;">
                <strong>P<sub>max</sub> = 3F / (2&pi;a<sup>2</sup>)</strong>
            </p>
            <p>Where:</p>
            <ul style="margin-bottom: 0;">
                <li><strong>F</strong> = Normal Load [N]</li>
                <li><strong>a</strong> = Contact radius [mm] = <sup>3</sup>&radic;(3FR / 4E<sup>*</sup>)</li>
                <li><strong>R</strong> = Ball radius [mm] (<em>{config.BALL_RADIUS} mm</em>)</li>
                <li><strong>E<sup>*</sup></strong> = Effective elastic modulus [MPa] = E / (2 &times; (1 - &nu;<sup>2</sup>))</li>
            </ul>
            <p style="margin-top: 10px; font-style: italic; color: #555;">Material constants used: E = {config.E_MODULUS:g} MPa, &nu; = {config.POISSON_RATIO}</p>
        </div>

        <h2 id="sec2">2. Dataset descriptive statistics</h2>
        <div style="overflow-x:auto;">
            {desc_df.to_html(classes='table', border=0, float_format=lambda x: '%.3f' % x)}
        </div>
"""

    if distribution_summary is not None:
        dist_html = distribution_summary.to_html(classes='table', border=0, index=False)
        html_content += f"""
        <h3>File Distribution by Operating Conditions</h3>
        <div style="overflow-x:auto; max-height: 400px; margin-bottom: 30px;">
            {dist_html}
        </div>"""

    if plotly_3d_html:
        html_content += f"""
        <h3>Interactive 3D Data Distribution</h3>
        <p><em>Use your mouse to rotate, zoom, and pan the 3D plot below.</em></p>
        <div class="no-print" style="margin-bottom: 30px; border: 1px solid #ccc; padding: 10px; background-color: #fafafa;">
            {plotly_3d_html}
        </div>"""

    html_content += """
        <h2 id="sec3">3. Model results comparison</h2>
        <div style="overflow-x:auto;">
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>R² (Train/Test/CV)</th>
                        <th>R² (COF/FAI)</th>
                        <th>RMSE (Train/Test)</th>
                        <th>RMSE (COF/FAI)</th>
                        <th>MAE (Test)</th>
                        <th>Optimum (Conc./Load/Temp.)</th>
                        <th>Predicted value at the last 5 minutes (COF/FAI)</th>
                        <th>Run-in Time [s]</th>
                        <th>Tuning & Training Time/Prediction Time</th>
                    </tr>
                </thead>
                <tbody>
    """
                
    for res in sorted_results:
        opt_str = f"{res['Opt_Conc']:.2f}/{int(res['Opt_Load'])}/{int(res['Opt_Temp'])}"
        pred_str = f"{res['Pred_COF']:.3f}/{res['Pred_FAI']:.3f}"
        time_str = f"{format_time(res['Tuning_Training_Time'])}/{res['Pred_Time_ms']:.1f}ms"
        r2_str = f"{res['R2_Train']:.2f}/{res['R2_Test']:.2f}/{res['R2_CV']:.2f}"
        r2_split_str = f"{res['R2_COF']:.2f}/{res['R2_FAI']:.2f}"
        rmse_str = f"{res['RMSE_Train']:.4f}/{res['RMSE_Test']:.4f}"
        rmse_split_str = f"{res['RMSE_COF']:.4f}/{res['RMSE_FAI']:.4f}"
        
        runin_val = res.get('RunIn_Time', 0)
        runin_str = f"{runin_val:.1f}" if isinstance(runin_val, (int, float)) else "N/A"
        
        row_style = ' style="font-weight: bold;"' if res['Name'] == best_model_res['Name'] else ''
        
        html_content += f"""
                    <tr{row_style}>
                        <td><strong>{res['Name']}</strong></td>
                        <td>{r2_str}</td>
                        <td>{r2_split_str}</td>
                        <td>{rmse_str}</td>
                        <td>{rmse_split_str}</td>
                        <td>{res['MAE_Test']:.4f}</td>
                        <td>{opt_str}</td>
                        <td>{pred_str}</td>
                        <td>{runin_str}</td>
                        <td>{time_str}</td>
                    </tr>"""
                    
    html_content += """
                </tbody>
            </table>
        </div>

        <h2 id="sec4">4. Detailed model results</h2>"""

    for res in sorted_results:
        feat_imp_str = "N/A"
        if res['Feature_Imp'] is not None:
            feats = [(config.NAME_MAPPING.get(f, f), val) for f, val in zip(res['Selected_Features'], res['Feature_Imp'])]
            feats.sort(key=lambda x: x[1], reverse=True)
            
            feat_imp_str = '<table class="feat-table"><thead><tr><th>Feature</th><th>Importance</th></tr></thead><tbody>'
            for fn, val in feats:
                feat_imp_str += f"<tr><td>{fn}</td><td>{val:.4f}</td></tr>"
            feat_imp_str += "</tbody></table>"
            
        params_str = "<ul>" + "".join([f"<li>{k}: {v}</li>" for k, v in res['Best_Params'].items()]) + "</ul>" if isinstance(res['Best_Params'], dict) else str(res['Best_Params'])
        
        html_content += f"""
        <div class="details-box">
            <h3>{res['Name']}</h3>
            <ul>
                <li><strong>R² (Train/Test/CV):</strong> {res['R2_Train']:.4f}/{res['R2_Test']:.4f}/{res['R2_CV']:.4f}</li>
                <li><strong>R² Split (COF/FAI):</strong> {res['R2_COF']:.4f}/{res['R2_FAI']:.4f}</li>
                <li><strong>RMSE (Train/Test):</strong> {res['RMSE_Train']:.4f}/{res['RMSE_Test']:.4f}</li>
                <li><strong>RMSE Split (COF/FAI):</strong> {res['RMSE_COF']:.4f}/{res['RMSE_FAI']:.4f}</li>
                <li><strong>MAE (Test):</strong> {res['MAE_Test']:.4f}</li>
                <li><strong>Best hyperparameters:</strong> {params_str}</li>
                <li><strong>Times (Tuning & Train / Predict):</strong> {format_time(res['Tuning_Training_Time'])} / {res['Pred_Time_ms']:.2f} ms</li>
            </ul>
            <strong>Feature importance:</strong><br>
            {feat_imp_str}
        """
        
        opt_curve_img = res.get('Opt_Curve_File', '')
        if opt_curve_img:
            html_content += f"""
            <div style="margin-top:15px;">
                <strong>Optimum Curve (vs Base Oil):</strong><br>
                <a href="{opt_curve_img}" target="_blank">
                    <img src="{opt_curve_img}" alt="{res['Name']} Optimum Curve" style="max-width:600px; margin-top:10px; border:1px solid #ccc;">
                </a>
                <p style="font-size: 0.9em; color: #666; font-style: italic;">{desc_map.get(opt_curve_img, '')}</p>
            </div>"""

        html_content += "</div>"

    html_content += """
        <h2 id="sec5">5. Generated diagrams</h2>
        <div class="img-grid">"""

    png_files = sorted(glob.glob(os.path.join(results_dir, "*.png")))
    for png_path in png_files:
        filename = os.path.basename(png_path)
        if "optimum_curve_" in filename:
            continue
        description = desc_map.get(filename, "Predicted curve or analysis plot.")
        
        html_content += f"""
            <div class="img-card">
                <h3>{filename.replace('.png', '').replace('_', ' ')}</h3>
                <a href="{filename}" target="_blank">
                    <img src="{filename}" alt="{filename}">
                </a>
                <p style="font-size: 0.9em; color: #666; font-style: italic;">{description}</p>
            </div>"""

    html_content += f"""
        </div>

        <h2 id="sec6">6. Evaluation and Optimum Comparison</h2>
        <div class="details-box">
            <p>Based on the investigation, the best performing model is: <strong>{best_model_res['Name']}</strong>.</p>
            <ul>
                <li><strong>R² accuracy (Test):</strong> {best_model_res['R2_Test']:.4f}</li>
                <li><strong>RMSE error:</strong> {best_model_res['RMSE_Test']:.4f}</li>
            </ul>
            <p><strong>Limitations & Generalization Gap:</strong><br>
            The observed difference between training and testing R² scores (Generalization gap) suggests the presence of latent physical variables that vary between files but are not captured by the model (e.g., microscopic surface roughness variations, slight sensor calibration drift, or humidity fluctuations). While the model captures the main trends, these unmeasured factors introduce irreducible error.</p>
            
            {f'<div style="margin-top:15px; border-top:1px solid #ccc; padding-top:10px;"><strong>SHAP Analysis (Impact of Physical Parameters):</strong><br>{shap_text}</div>' if shap_text else ''}
            
            <h3>Not esterified vs. Esterified Oil Optimums</h3>
            <table>
                <thead>
                    <tr>
                        <th>Type</th>
                        <th>Opt. Concentration [%]</th>
                        <th>Opt. Load [N]</th>
                        <th>Opt. Temperature [°C]</th>
                        <th>Expected COF</th>
                        <th>Expected FAI</th>
                        <th>Run-in Time</th>
                        <th>Stability</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Not esterified (0)</strong></td>
                        <td>{optimum_results[0]['Conc']:.2f}</td>
                        <td>{int(optimum_results[0]['Load'])}</td>
                        <td>{int(optimum_results[0]['Temp'])}</td>
                        <td>{optimum_results[0]['COF']:.4f}</td>
                        <td>{optimum_results[0]['FAI']:.3f}</td>
                        <td>{optimum_results[0].get('RunInStr', 'N/A')}</td>
                        <td>{optimum_results[0].get('Stability', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td><strong>Esterified (1)</strong></td>
                        <td>{optimum_results[1]['Conc']:.2f}</td>
                        <td>{int(optimum_results[1]['Load'])}</td>
                        <td>{int(optimum_results[1]['Temp'])}</td>
                        <td>{optimum_results[1]['COF']:.4f}</td>
                        <td>{optimum_results[1]['FAI']:.3f}</td>
                        <td>{optimum_results[1].get('RunInStr', 'N/A')}</td>
                        <td>{optimum_results[1].get('Stability', 'N/A')}</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <h2 id="sec7">7. DoE Suggestions (New measurements)</h2>
        <div class="details-box">
            <p>Suggestions are based on a combination of model uncertainty (Bagging variance) and distance from existing measurements (Sparsity). The goal is to investigate uncertain and unexplored areas.</p>
            <table>
                <thead>
                    <tr>
                        <th>Oil Type</th>
                        <th>Concentration [%]</th>
                        <th>Load [N]</th>
                        <th>Temperature [°C]</th>
                        <th>Uncertainty (Std Dev)</th>
                        <th>Distance (Sparsity)</th>
                        <th>Combined Score</th>
                    </tr>
                </thead>
                <tbody>"""
                
    for _, row in doe_suggestions.iterrows():
        oil_type = "Esterified" if row['Esterified'] == 1 else "Not esterified"
        html_content += f"""
                    <tr>
                        <td>{oil_type}</td>
                        <td>{row['Concentration']:.2f}</td>
                        <td>{int(row['Load'])}</td>
                        <td>{int(row['Temperature'])}</td>
                        <td>{row['Uncertainty_COF']:.4f}</td>
                        <td>{row['Distance']:.4f}</td>
                        <td>{row['Score']:.4f}</td>
                    </tr>"""
                    
    html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Created by: Kevin Szabó</p>
        </div>
    </div>
</body>
</html>"""
    
    return html_content
