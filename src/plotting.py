import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from matplotlib import cm


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
                               xticklabels=2):
    """Followed [this](https://seaborn.pydata.org/examples/many_pairwise_correlations.html) example"""
    df = pd.DataFrame(X, columns=column_names)
    df['magnitude'] = y

    corr = df.corr()

    sns.set_theme(style="white")

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

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

def plot_nested_rfecv_boxplots(results_dict):
    fig, ax = plt.subplots(2, 1)
    ax, ax2 = ax
    # ax2 = ax.twinx()

    xlabels = []
    for i, key in enumerate(results_dict.keys()):
        xlabels.append(key)
        ts_optfts = results_dict[key]['test_score_optfts']
        ts_allfts = results_dict[key]['test_score_allfts']

        median1, q1, q3, if1, if2 = get_boxplot_values(ts_optfts)
        # ax.scatter(i, median, color='C0', marker='x', s=20)
        ax.vlines(x=i, ymin=q1, ymax=q3, alpha=0.5, lw=5, color='C0')
        ax.vlines(x=i, ymin=if1, ymax=if2, alpha=0.5, lw=1, color='C0')

        median2, q1, q3, if1, if2 = get_boxplot_values(ts_allfts)
        ax.vlines(x=i, ymin=q1, ymax=q3, alpha=0.5, lw=5, color='C1')
        ax.vlines(x=i, ymin=if1, ymax=if2, alpha=0.5, lw=1, color='C1')

        label1, label2, = None, None
        if i == 1:
            label1 = 'selected'
            label2 = 'all'
        ax.scatter(i, median1, color='C0', marker='x', s=20, zorder=5, label=label1)
        ax.scatter(i, median2, color='C1', marker='x', s=20, zorder=5, alpha=0.7, label=label2)

        n_feats = results_dict[key]['n_feats']
        median, q1, q3, if1, if2 = get_boxplot_values(n_feats)
        ax2.scatter(i, median, color='C3', marker='x', s=20)
        ax2.vlines(x=i, ymin=q1, ymax=q3, alpha=0.5, lw=5, color='C3')
        ax2.vlines(x=i, ymin=if1, ymax=if2, alpha=0.5, lw=1, color='C3')
        
    ax.set_xticks(np.arange(len(xlabels)), labels=[])
    ax2.set_xticks(np.arange(len(xlabels)), labels=xlabels, rotation=90);
    ax.set_ylabel(r'CV $R^2$')
    ax2.set_ylabel(r'CV N Features')
    ax.legend()

def plot_rfecv_feature_heatmap(mega_df, feature_names):
    fig, ax = plt.subplots()
    mappable = ax.imshow(mega_df.to_numpy(), cmap=cm.Blues)
    ax.set_yticks(np.arange(mega_df.shape[0]), feature_names);
    ax.set_xticks(np.arange(mega_df.shape[1]), mega_df.columns, rotation=90);
    fig.colorbar(mappable, shrink=0.6, label='CV Count')

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
                                title=None):
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

    ax.set_xticks(df['station']);
    ax.set_xticklabels(df['station'], rotation=90);
    ax.set_ylabel(f'${ylabel}$')
    ax.legend()
    ax.set_title(title)