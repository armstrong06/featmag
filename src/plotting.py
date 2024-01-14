import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class FeaturePlots():
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
        df = pd.DataFrame(X, columns=column_names)
        df['magnitude'] = y

        corr = df.corr()

        sns.set_theme(style="white")

        # Generate a mask for the upper triangle
        mask = np.trim(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5},
                    yticklabels=1, xticklabels=xticklabels).set_title(title)
