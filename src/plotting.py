import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from matplotlib import patches
from matplotlib import cm
from matplotlib.colors import Normalize, TwoSlopeNorm
from sklearn.metrics import r2_score, mean_squared_error
import string

def plot_station_feature_box_whisker(df, feature, ylabel=None, sort_counts=True, thresholds=None, showfliers=True, min_y_line=None):
    fig = plt.figure(figsize=(15, 5))
    station_feat = df.groupby("station").apply(
        lambda x: x[feature].values).to_frame()
    station_feat.columns = ["values"]
    if sort_counts:
        station_feat.loc[:, "counts"] = station_feat.apply(
            lambda x: len(x["values"]), axis=1)
        station_feat = station_feat.sort_values("counts", ascending=False)

    if thresholds is not None:
        for thresh in thresholds:
            thresh_ind = np.where(station_feat.counts < thresh)[0][0] + 0.5
            plt.axvline(thresh_ind, color="red", alpha=0.5)

    if min_y_line is not None:
        plt.axhline(min_y_line, color='r')
    labels = station_feat.index
    plt.boxplot(station_feat["values"].to_numpy(), showfliers=showfliers)
    plt.xticks(range(1, len(labels)+1), labels=labels, rotation="vertical")
    if ylabel is None:
        ylabel = feature
    plt.ylabel(ylabel)


def plot_pairwise_correlations(X, y, column_names, title,
                               xticklabels=2, ax=None):
    """Followed [this](https://seaborn.pydata.org/examples/many_pairwise_correlations.html) example"""
    df = pd.DataFrame(X, columns=column_names)
    df['magnitude'] = y

    corr = df.corr()

    sns.set_theme(style="white")

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    if ax is None:
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                yticklabels=1, xticklabels=xticklabels).set_title(title)


def plot_r2_heatmap(gs_results,
                    C_range,
                    gamma_range,
                    opt_ind,
                    station,
                    outdir=None,
                    show=True):
    scores = gs_results.cv_results_["mean_test_score"].reshape(
        len(C_range), len(gamma_range))
    fig, ax = plt.subplots()
    mappable = ax.imshow(scores,
                         interpolation="nearest",
                         cmap=plt.cm.hot,
                         )
    best_ind = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
    opt_ind2d = np.unravel_index(opt_ind, scores.shape)
    ax.scatter(best_ind[1], best_ind[0], marker='*',
               color='k', label='Best Model')
    ax.scatter(opt_ind2d[1], opt_ind2d[0], marker='*',
               color='blue', label='Selected Model')
    ax.set_xlabel("log gamma")
    ax.set_ylabel("log C")
    fig.colorbar(mappable, label=r"Mean $R^2$")
    ax.set_xticks(np.arange(len(gamma_range)),
                  np.log10(gamma_range), rotation=45)
    ax.set_yticks(np.arange(len(C_range)), np.log10(C_range))
    ax.set_title(f'{station} CV Results')
    ax.legend(handletextpad=0.2, borderpad=0.2, handlelength=1.0)
    if outdir is not None:
        fig.savefig(os.path.join(
            outdir, f'{station}.r2.cvheatmap.png'), dpi=300)
        
    if show:
        fig.show()
    else:
        plt.close()

def plot_intrinsic_score_box_whisker(dists, title, xlabels, ylabel, sort_col_inds=None):
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_title(title)
    if sort_col_inds is not None:
        dists = dists[:, sort_col_inds]
        xlabels = np.array(xlabels)[sort_col_inds]
    ax1.boxplot(dists);
    ax1.set_xticks(np.arange(1, len(xlabels)+1), labels=xlabels, rotation=90, fontsize=8);
    ax1.set_ylabel(ylabel)

def plot_station_intrinsic_score_bar_chart(scores, stat_order, stat, xlabels, ylabel):
    fig = plt.figure(figsize=(8, 4))
    s_scores = scores[:, np.where(stat_order == stat)[0][0]]
    plt.title(stat)
    plt.bar([i for i in range(len(s_scores))], s_scores)
    plt.xlim([-1, len(s_scores)])
    plt.xticks(np.arange(len(s_scores)), labels=xlabels, rotation=90, fontsize=8)
    plt.ylabel(ylabel)
    plt.show()

# From this example https://matplotlib.org/stable/gallery/statistics/customized_violin.html#sphx-glr-gallery-statistics-customized-violin-py
def get_boxplot_values(vals):
    q1, median, q3 = np.percentile(vals, [25, 50, 75])
    ir = (q3 - q1)
    vals = np.sort(vals)
    # Upper inner fence
    upper_inner_fence = q3 +  ir * 1.5
    upper_adjacent_value = np.clip(upper_inner_fence, q3, vals[-1])
    # Lower inner fence
    lower_inner_fence = q1 - ir * 1.5
    lower_adjacent_value = np.clip(lower_inner_fence, vals[0], q1)

    return median, q1, q3, lower_adjacent_value, upper_adjacent_value

def plot_rfecv_score_summary(rfecv_results_dict, rfe_results_dict,
                               capsize=5,
                               score_ylims=None,
                               grids=False,
                               rfecv_boxplot=False,
                               best_sym = 'o',
                               oste_sym='x',
                               s=20,
                               elinewidth=None,
                               plot_N=False,
                               N_ylims=None):
    if not plot_N:
        fig, axes = plt.subplots(2, 1, figsize=(5, 5))
        ax_rfecv, ax_rfe = axes
    else:
        fig, axes = plt.subplots(3, 1, figsize=(7, 5))
        ax_N, ax_rfecv, ax_rfe = axes
    # ax2 = ax.twinx()
    results_df = pd.DataFrame(rfecv_results_dict).T.reset_index().rename(columns={'index':'station'}).sort_values('station')
    xlabels = results_df['station']
    for i, row in results_df.iterrows():
        scores = np.array(row['N_scores'])
        best_N = row['best_N']
        oste_N = row['oste_N']
        label1, label2, = None, None
        if i == 1:
            label1 = 'best $\it{N}$'
            label2 = '1 st. err. $\it{N}$'

        if plot_N:
            ax_N.scatter(i, best_N, color='C0', marker=best_sym, s=s, zorder=5)
            ax_N.scatter(i, oste_N, color='C1', marker=oste_sym, s=s, zorder=5)

        ts_bestfeats = scores[:, best_N-1]
        if rfecv_boxplot:
            median1, q1, q3, if1, if2 = get_boxplot_values(ts_bestfeats)
            # ax.scatter(i, median, color='C0', marker='x', s=20)
            ax_rfecv.vlines(x=i, ymin=q1, ymax=q3, alpha=0.5, lw=5, color='C0')
            ax_rfecv.vlines(x=i, ymin=if1, ymax=if2, alpha=0.5, lw=1, color='C0')
            ax_rfecv.scatter(i, median1, color='C0', marker=best_sym, s=s, zorder=5)
            ts_ostefeats, median2 = None, None
            if best_N != oste_N:
                ts_ostefeats = scores[:, row['oste_N']-1]
                median2, q1, q3, if1, if2 = get_boxplot_values(ts_ostefeats)
                ax_rfecv.vlines(x=i, ymin=q1, ymax=q3, alpha=0.5, lw=5, color='C1')
                ax_rfecv.vlines(x=i, ymin=if1, ymax=if2, alpha=0.5, lw=1, color='C1')
                ax_rfecv.scatter(i, median2, color='C1', marker=oste_sym, s=s, zorder=5, alpha=0.7, label=label2)
        else:
            ax_rfecv.errorbar(i, np.mean(ts_bestfeats), np.std(ts_bestfeats), color='C0',
                     capsize=capsize,
                     alpha=0.5,
                     elinewidth=elinewidth)
            ax_rfecv.scatter(i, np.mean(ts_bestfeats), color='C0', marker=best_sym, s=s,
                              zorder=100)
            if best_N != oste_N:
                ts_ostefeats = scores[:, row['oste_N']-1]
                ax_rfecv.errorbar(i, np.mean(ts_ostefeats), np.std(ts_ostefeats), color='C1',
                     capsize=capsize,
                     alpha=0.5, 
                     elinewidth=elinewidth)
                ax_rfecv.scatter(i, np.mean(ts_ostefeats), color='C1', marker=oste_sym, s=s,
                           zorder=100)

        stat = row['station']
        rfe_best_mean = rfe_results_dict[stat]['best']['pred_cv_mean']
        rfe_best_std = rfe_results_dict[stat]['best']['pred_cv_std']
        ax_rfe.errorbar(i, rfe_best_mean, rfe_best_std, color='C0',
                     capsize=capsize,
                     alpha=0.5, 
                     elinewidth=elinewidth)
        ax_rfe.scatter(i, rfe_best_mean, color='C0', marker=best_sym, s=s, zorder=100,
                       label=label1)
        rfe_oste_dict =  rfe_results_dict[stat]['oste']
        if rfe_oste_dict is not None:
            rfe_oste_mean = rfe_results_dict[stat]['oste']['pred_cv_mean']
            rfe_oste_std = rfe_results_dict[stat]['oste']['pred_cv_std']
            ax_rfe.errorbar(i, rfe_oste_mean, rfe_oste_std, color='C1',
                         capsize=capsize, alpha=0.5, 
                         elinewidth=elinewidth)
            ax_rfe.scatter(i, rfe_oste_mean, color='C1', marker=oste_sym, s=s, zorder=100,
                           label=label2)

    if plot_N:
        ax_N.set_ylabel("$\it{N}$ Features", fontsize=9)
        ax_N.set_xticks(np.arange(len(xlabels)), labels=[])
        ax_N.set_ylim(N_ylims)
        if grids:
            ax_N.grid(axis='y')
    ax_rfe.legend(fontsize=8)
    ax_rfecv.set_xticks(np.arange(len(xlabels)), labels=[])
    ax_rfe.set_xticks(np.arange(len(xlabels)), 
                      labels=xlabels, 
                      rotation=90,
                      fontsize=8);
    ax_rfecv.set_ylabel(r'RFECV $R^2$', fontsize=9)
    ax_rfe.set_ylabel(r'Selected Feats. CV $R^2$', fontsize=9)
    ax_rfecv.set_ylim(score_ylims)
    ax_rfe.set_ylim(score_ylims)
    if grids:
        ax_rfecv.grid(axis='y')
        ax_rfe.grid(axis='y')

    for ax in axes:
        for item in ax.get_yticklabels():
            item.set_fontsize(8)

def plot_rfecv_feature_heatmap(mega_df, 
                               ax=None,
                               plot_colorbar=True,
                               title=None,
                               fontsize=8,
                               figsize=None):
    feature_names = mega_df.index.values
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    mappable = ax.imshow(mega_df.to_numpy(), cmap=cm.Blues)
    ax.set_yticks(np.arange(mega_df.shape[0]), 
                  feature_names,
                  fontsize=fontsize);
    ax.set_xticks(np.arange(mega_df.shape[1]), 
                  mega_df.columns, 
                  rotation=90,
                  fontsize=fontsize);
    if plot_colorbar:
        plt.colorbar(mappable,
                      shrink=0.6).set_label(label='CV Count',
                                            fontsize=fontsize+1) 
    ax.set_title(title, fontsize=fontsize+1)
    return mappable

def compare_score_different_feats_scatter(df1,
                                   df2,
                                   df1_label,
                                   df2_label,
                                   col_name='test_r2',
                                   ylabel='R^2',
                                   title=None):
    df1_color = '#1b9e77'
    df2_color = '#d95f02'

    merge_df = df1[['station', col_name]].merge(df2[['station', col_name]], 
                                                on='station', 
                                                how='outer')
    print(merge_df.columns)

    fig, ax = plt.subplots()
    ax.scatter(merge_df['station'], 
            merge_df[f'{col_name}_x'],
            marker='*', 
            color=df1_color, 
            label=df1_label,
            s=50)
    
    ax.scatter(merge_df['station'], 
            merge_df[f'{col_name}_y'],
            marker='x', 
            color=df2_color, 
            label=df2_label)

    ax.set_xticks(merge_df['station']);
    ax.set_xticklabels(merge_df['station'], rotation=90);
    ax.set_ylabel(f'${ylabel}$')
    ax.legend(loc='lower left')
    ax.set_title(title)

def plot_station_splits_scores_scatter(df,
                                metric='r2',
                                ylabel='R^2',
                                title=None,
                                vlines=False):
    train_color = '#1b9e77'
    test_color = '#d95f02'
    holdout_color = '#7570b3'

    fig, ax = plt.subplots()
    if metric == 'r2':
        ax.vlines(df['station'], 
                df['cv_mean_sel']-df['cv_std_sel'],
                df['cv_mean_sel']+df['cv_std_sel'],
                lw=4,
                color=train_color,
                alpha=0.5,
                zorder=0,
                label='CV St. Dev.')
        ax.scatter(df['station'], 
                df['cv_mean_sel'], 
                marker='_', 
                color=train_color,
                label='CV Mean')
    ax.scatter(df['station'], 
            df[f'train_{metric}'],
            marker='x', 
            color=train_color, 
            label='Train Score',
            zorder=4)
    ax.scatter(df['station'], 
            df[f'holdout_{metric}'], 
            marker='^', 
            color=holdout_color,
            label='Holdout Score',
            zorder=3,
            s=55)
    ax.scatter(df['station'], 
            df[f'test_{metric}'], 
            marker='*', 
            color=test_color,
            label='Test Score',
            s=50,
            zorder=5)
    if vlines:
        ax.vlines(df['station'], 
                df[f'holdout_{metric}'], 
                df[f'train_{metric}'], 
                color=holdout_color,
                alpha=0.5)
        ax.vlines(df['station'], 
                df[f'test_{metric}'], 
                df[f'train_{metric}'], 
                color=test_color,
                alpha=0.5)
    
    ax.set_xticks(df['station']);
    ax.set_xticklabels(df['station'], rotation=90);
    ax.set_ylabel(f'${ylabel}$')
    ax.legend()
    ax.set_title(title)

def scores_heatmap(df, 
                   cols=['train_r2', 'test_r2', 'holdout_r2'],
                   xticklabels=['Train', 'Test', 'Holdout'],
                   ax=None,
                   title=None,
                   show_ylabels=True,
                   show_cbar=True,
                   midpoint_normalize=True,
                   midpoint=None,
                   cmap=cm.Blues,
                   cmap_min=None,
                   cmap_max=None,
                   cbar_ticks=None,
                   cbar_label=None,
                   tablefontcolor='k',
                   tablefontsize=None,
                   tight_layout=True,
                   tablevalueprec=2):
    """Follow this example
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#a-simple-categorical-heatmap
    """
    norm = None
    if midpoint_normalize:
        if midpoint is None:
            midpoint = np.nanpercentile(df[cols].values, [50])[0]
            print(f'cmap midpoint set to {midpoint}')
        if cmap_min is None:
            cmap_min = df[cols].min(axis=None)
            print(f'cmap min set to {cmap_min}')
        if cmap_max is None:
            cmap_max = df[cols].max(axis=None)
            print(f'cmap max set to {cmap_max}')

        norm = TwoSlopeNorm(vmin=cmap_min, vcenter=midpoint, vmax=cmap_max)
         
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 6))

    im = ax.imshow(df[cols], 
                aspect='auto',
                norm=norm,
                cmap=cmap,
                #clim=(cmap_min, cmap_max)                     
                )
    ylabels=[]
    yticks = []
    if show_ylabels:
        ylabels=df['station']
        yticks = np.arange(len(df['station']))
    ax.set_yticks(yticks, labels=ylabels);
    ax.set_xticks(np.arange(len(cols)), xticklabels);
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor");
    # Loop over data dimensions and create text annotations.
    for i in range(len(df['station'])):
        for j in range(3):
            t = df[cols].values[i, j]
            if np.isnan(t):
                t=""
            else:
                t=f"{t:0.{tablevalueprec}f}"
            ax.text(j, i, t,
                    ha="center", 
                    va="center", 
                    color=tablefontcolor,
                    fontsize=tablefontsize)
    if show_cbar:   
        plt.colorbar(im, ticks=cbar_ticks, label=cbar_label)
    
    ax.set_title(title)

    if tight_layout and (ax is None):
        fig.tight_layout()

    return im

def actual_v_predicted(results_df,
                         all_train_df,
                         all_test_df,
                         all_holdout_df = None,
                         n_cols=6,
                         title=None,
                         tick_locations=[0, 2, 4],
                         inner_ticks_on=True,
                         axis_lims=[-0.2, 5],
                         legend_names=['Train', 'Test', 'Holdout'],
                         colors = ['lightgray', 'C0', 'C1'],
                         alphas = [0.5, 0.5, 0.5],
                         legend_bbox_width=3,
                         figsize=(11,11),
                         linestyle='-',
                         linecolor='k'):

    inner_tick_locations = []
    if inner_ticks_on:
        inner_tick_locations = tick_locations

    # Set up the figure axes
    n_stats = len(results_df)
    n_rows = int(np.ceil((n_stats)/n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    # When the number of stations does not evenly fit into the n_cols*n_rows,
    # remove extra axes from the upper left corner 
    # Get the ax inds to turn off and add 'null' into the station list
    stat_list = results_df['station'].sort_values().tolist()
    ignore_inds = []
    for i in range(int(n_rows*n_cols - n_stats), 0, -1): 
        ignore_inds.append(n_cols-i)
        stat_list.insert(n_cols-i, "null")

    subpanel_labels = list(string.ascii_lowercase)
    i = 0
    legend = False
    for cnt, station in enumerate(stat_list):
        ax = axes[cnt]
        ax.set_xlim(axis_lims)
        ax.set_ylim(axis_lims)
        # Turn of the extra axes
        if cnt in ignore_inds:
            if not legend:
                legend_symbol_x = 1.0
                legend_label_x = legend_symbol_x + 0.35
                legend_symbol_max_y = 3.05
                legend_symbol_spacing_y = 0.75
                legend_label_shift_y = 0.15
                bbox_pad = 0.2
               # bbox_x_length = 3
                for legend_i in range(3):
                    symbol_pos_y = legend_symbol_max_y-legend_i*legend_symbol_spacing_y
                    text_pos_y = symbol_pos_y - legend_label_shift_y
                    ax.scatter(legend_symbol_x, symbol_pos_y, 
                            color=colors[legend_i], 
                            alpha=alphas[legend_i])
                    ax.text(legend_label_x, text_pos_y, legend_names[legend_i], fontsize=12)

                rect = patches.Rectangle((legend_symbol_x-2*bbox_pad, text_pos_y-bbox_pad),
                                         width=legend_bbox_width,
                                         height=(legend_symbol_max_y+3*bbox_pad)-text_pos_y,
                                         linewidth=1,
                                         edgecolor='k',
                                         facecolor='none')
                ax.add_patch(rect)

                legend = True
                ax.plot([])
            ax.axis('off')
            continue

        # Get the subpanel label
        sp_label = subpanel_labels[i%len(subpanel_labels)]
        if i >= len(subpanel_labels):
            sp_label = subpanel_labels[0] + sp_label
        sp_label = "(" + sp_label + ")"
        # Get the predictions for one station
        results_row = results_df[results_df['station'] == station]
        train_df = all_train_df[all_train_df['station'] == station]
        test_df = all_test_df[all_test_df['station'] == station]
        holdout_df = None
        if (all_holdout_df is not None) and (station in all_holdout_df['station'].unique()):
            holdout_df = all_holdout_df[all_holdout_df['station'] == station]

        # Set ticks & labels - axes on the outer edge should have labels
        if cnt < len(stat_list) - n_cols:
            ax.set_xticks(inner_tick_locations, labels=[])
        else:
            ax.set_xticks(tick_locations)
        if (cnt % n_cols != 0):
            ax.set_yticks(inner_tick_locations, labels=[])

        ax.scatter(train_df['magnitude'], 
                train_df['predicted_magnitude'], 
                label=legend_names[0], 
                color=colors[0], 
                alpha=alphas[0])

        ax.scatter(test_df['magnitude'], 
                test_df['predicted_magnitude'], 
                label=legend_names[1], 
                color=colors[1], 
                alpha=alphas[1])
        
        if holdout_df is not None:
            ax.scatter(holdout_df['magnitude'], 
                    holdout_df['predicted_magnitude'], 
                    label=legend_names[2], 
                    color=colors[2], 
                    alpha=alphas[2])
            
        ax.plot(np.arange(axis_lims[0], axis_lims[1], 0.5), 
                np.arange(axis_lims[0], axis_lims[1], 0.5), 
                color=linecolor,
                linestyle=linestyle)
        ax.text(1.5, 
                4.2, 
                station, 
                fontsize=14) #, bbox={"facecolor":"white", "alpha":0.5}

        ax.text(0.05, 
                1.03, 
                sp_label, 
                transform=ax.transAxes,
                fontsize=12)

        # We change the fontsize of minor ticks label 
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=8)

        ax.text(1.8, 0.01, 
                f"{results_row['test_r2'].values[0]:1.2f}", 
                color=colors[1], 
                fontsize=12)
        if holdout_df is not None:
            ax.text(3.2, 0.01, 
                    f"{results_row['holdout_r2'].values[0]:1.2f}", 
                    color=colors[2],
                    fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='white', pad=0.1))

        ax.set_aspect('equal', adjustable='box')
        i+=1
        
    fig.supxlabel(r"Actual $M_L$", fontsize=16)
    fig.supylabel(r"Predicted $M_L$", fontsize=16, x=-0.02)
    fig.suptitle(title, fontsize=16, y=1.04)

    if not legend:
        ax.legend(loc=(1.2, 0), fontsize=12, handletextpad=0.5, borderpad=0.05, 
                borderaxespad=0.05, handlelength=0.5)
    

def actual_v_network_avg_prediction(df_list,
                                    title = None,
                                    plot_xlabel=True,
                                    plot_ylabel=True,
                                    plot_lims=[-0.5, 4.0],
                                    plot_ytick_labels=True,
                                    plot_xtick_labels=True,
                                    ax=None,
                                    alphas=[0.1],
                                    legend_labels=[None],
                                    plot_legend=False,
                                    marker_colors=['C0'],
                                    linecolor='k',
                                    linestyle='-',
                                    text_x=[0.0],
                                    text_y=[-1.2]
                                    ):

    if ax is None:
        fig, ax = plt.subplots(1, constrained_layout=True)

    yticks = np.arange(0, plot_lims[1]+0.5, 1, dtype=int)
    ytick_labels = yticks
    xtick_labels = yticks

    if not plot_ytick_labels:
        ytick_labels = []
    if not plot_xtick_labels:
        xtick_labels = []

    for i, df in enumerate(df_list):
        r2 = r2_score(df['magnitude'], df['predicted_magnitude'])
        rmse = mean_squared_error(df['magnitude'], 
                                  df['predicted_magnitude'], 
                                  squared=False)
        ax.scatter(df["magnitude"], 
                   df["predicted_magnitude"], 
                   alpha=alphas[i],
                   label=legend_labels[i],
                   color=marker_colors[i])
        ax.text(text_x[i] + i*1.0, plot_lims[1]+text_y[i],
                f"$N$:{df.shape[0]: 0.0f}\n$R^2$:{r2: 0.2f}\n$RMSE$:{rmse: 0.2f}", 
                fontsize=10,
                color=marker_colors[i],
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='white')) 


        # ax.text(text_start[i]+0.2+ i*1.5, plot_lims[1]-0.475, 
        #         f"$N$={df.shape[0]: 0.0f}", fontsize=12,
        #         color=marker_colors[i])
        # ax.text(text_start[i]+0.1 + i*1.5, plot_lims[1]-0.75, 
        #         f"$R^2$={r2: 0.2f}", fontsize=12,
        #         color=marker_colors[i])
        # ax.text(text_start[i]+-0.2+ i*1.5, plot_lims[1]-1.0, 
        #         f"$RMSE$={rmse: 0.2f}", fontsize=12,
        #         color=marker_colors[i])

    ax.plot(np.arange(plot_lims[0], plot_lims[1]+0.5, 0.5), 
            np.arange(plot_lims[0], plot_lims[1]+0.5, 0.5), 
            color=linecolor,
            linestyle=linestyle)

    if plot_xlabel:
        ax.set_xlabel(r"Actual $M_L$", fontsize=12)
    if plot_ylabel:
        ax.set_ylabel(r"Predicted Network (Average) $M_L$", fontsize=12)
    if plot_legend:
        ax.legend(loc='lower right')

    ax.set_title(title, fontsize=12)
    ax.set_yticks(yticks, labels=ytick_labels);
    ax.set_xticks(yticks, labels=xtick_labels);
    ax.set_ylim(plot_lims);
    ax.set_xlim(plot_lims)
    ax.set_aspect('equal', adjustable='box')

def plot_sequential_all_feature_scores(all_scores, initial_score=0.0):
    feat_max = np.insert(np.nanmax(all_scores[:, :, 0], axis=1), 0, initial_score)
    feat_min = np.insert(np.nanmin(all_scores[:, :, 0], axis=1), 0, initial_score)
    feat_mean = np.insert(np.nanmean(all_scores[:, :, 0], axis=1), 0, initial_score)
    x = np.arange(feat_max.shape[0])
    plt.fill_between(x, feat_max, feat_min, color='gray', alpha=0.5)
    plt.plot(x, feat_mean, color='k', label='mean of all features', linestyle='--')
    plt.plot(x, feat_max, color='r', label='max feature', marker='x')
    plt.plot(x, feat_min, color='k', label='min feature')
    plt.xticks(x)
    plt.xlabel("N features")
    plt.ylabel("Mean CV $R^2$")
    plt.legend(loc='lower right')
    plt.title("Mean CV $R^2$ for Considered Features")

def plot_sequential_selected_feature_scores(ids_scores, 
                                            feature_names, 
                                            base_set_name="None"):
    feat_plot_names = np.concatenate([[base_set_name], feature_names])
    x = np.arange(ids_scores.shape[0])
    plt.fill_between(x, ids_scores[:, 1], ids_scores[:, 2], color='gray', alpha=0.5, label='CV min, max')
    plt.plot(x, ids_scores[:, 0], color='k', marker='x', label='CV mean')
    plt.xticks(x, feat_plot_names, rotation=90)
    plt.xlabel("Selected features")
    plt.ylabel("CV $R^2$")
    plt.legend(loc='lower right')
    plt.title("CV $R^2$ Ranges for Selected Features")
