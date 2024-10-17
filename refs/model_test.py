import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import numpy as np
import scipy.stats as stats
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



def featuresImportance(X, gmm):
    # Generate synthetic data with 3 features
    # X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, n_features=3, random_state=0)


    y_pred = gmm.predict(X)

    # Fit an ExtraTreesClassifier and use it to estimate feature importances
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X, y_pred)
    importances_tree = clf.feature_importances_

    # Fit a SelectKBest with chi2 and use it to estimate feature importances
    kbest = SelectKBest(score_func=chi2, k='all')
    fit = kbest.fit(X, y_pred)
    importances_kbest = fit.scores_
    index = ['avg_mapq', 'depth_cv', 'r0r_rate']
    # Create dataframes for visualization
    df_tree = pd.DataFrame(importances_tree, columns=['Importance'], index=X.columns)
    df_kbest = pd.DataFrame(importances_kbest, columns=['Importance'], index=X.columns)

    # Print the results
    print("Feature Importances Estimated by ExtraTreesClassifier:")
    print(df_tree)
    print("\nFeature Importances Estimated by SelectKBest with chi2:")
    print(df_kbest)

    # Plot the feature importances as estimated by ExtraTreesClassifier
    plt.figure(figsize=(6, 6), dpi=300)
    sns.barplot(x=df_tree.index, y=df_tree["Importance"], palette="viridis", width=0.2)
    plt.title("Feature Importances Estimated by ExtraTreesClassifier")
    plt.ylabel("Importance")
    # plt.xticks(rotation=45)
    sns.despine(top=True, right=True)
    plt.tick_params(axis='both', which='major', labelsize=6) 
    plt.savefig('featuresImportance1.png')

    # Plot the feature importances as estimated by SelectKBest with chi2
    plt.figure(figsize=(6, 6), dpi=300)
    sns.barplot(x=df_kbest.index, y=df_kbest["Importance"], palette="viridis", width=0.2)
    plt.title("Feature Importances Estimated by SelectKBest with chi2")
    plt.ylabel("Importance")
    # plt.xticks(rotation=5)
    sns.despine(top=True, right=True)
    plt.tick_params(axis='both', which='major', labelsize=6) 
    plt.savefig('featuresImportance.png')

def figure1(X):
    fig = plt.figure(figsize=(9, 9), dpi=300)
    gs = plt.GridSpec(4, 4)

    # Define axes
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_xDist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_yDist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    

    X = X[X['depth_mean'] < 1000]
    x = X['avg_mapq']
    
    y = X['depth_mean']
    # Create scatter plot on main ax
    ax_main.scatter(x, y, alpha=0.5, s=1)

    # Create histogram on the attached axes
    sns.kdeplot(x=x, fill=True, ax=ax_xDist, color='gray', lw=2)
    # ax_xDist.invert_xaxis()
    sns.kdeplot(y=y, fill=True, ax=ax_yDist, color='gray', lw=2)
    # ax_yDist.invert_yaxis()

    # Remove labels and ticks from the histogram axes
    # ax_xDist.set_xticklabels([])
    # ax_yDist.set_yticklabels([])
    ax_xDist.set_xlabel('')
    ax_yDist.set_ylabel('')

    # Label the main axes
    ax_main.set(xlabel="avg_mapq", ylabel="depth_mean")
    ax_main.xaxis.label.set_size(10)
    ax_main.yaxis.label.set_size(10)
    ax_main.xaxis.set_tick_params(labelsize=10)

    # ax_main.tick_params(axis='both', which='major', labelsize=10)

    # Set title for the figure
    # ax_main.set_title('Joint Probability with Marginals', size=20)
    sns.despine(ax=ax_main)
    sns.despine(ax=ax_xDist, top=True, right=True)
    sns.despine(ax=ax_yDist, top=True, right=True)
    plt.tight_layout()

    # Show the plot
    plt.savefig('avg_mapq_depth_mean.png')


def featureDistribution(X_df, feature):
    fig, ax = plt.subplots(dpi=300)
    # Plot the histogram of the data
    X=X_df[feature]

    # Suppose your data is in a numpy array X
    min_val = np.min(X)
    max_val = np.max(X)

    # Suppose you want each bin to be of size 5
    bin_size = 0.1
    # Calculate number of bins
    num_bins = np.ceil((max_val - min_val) / bin_size)
    print(num_bins)

    ax.hist(X, bins=int(num_bins), density=True, alpha=0.3, color='gray', label='Histogram')
    # Plot the KDE of the data
    sns.kdeplot(X, color='black', ax=ax, label='Kernel Density Estimation')
    # sns.kdeplot(X, color='black', ax=ax, label='Kernel Density Estimation', bw_adjust=0.5)

    # ax.hist(x0, bins=30, density=True, alpha=0.3, color='r', label='Data')
    # x= np.linspace(np.min(X), np.max(X), 1000)

    # x = X_scaled
    # x = np.linspace(X_df['r0r_rate'].min(), X_df['r0r_rate'].max(), 1000)
    # colors = ['g', 'r']
    # for i in range(gmm.n_components):
    #     pdf = norm.pdf(x, gmm.means_[i, 1], np.sqrt(gmm.covariances_[i, 1, 1])) * gmm.weights_[i]
    #     ax.plot(x, pdf, label=f'Component {i+1}')



    # Find the intersection point
    # def find_intersection(x, pdf1, pdf2):
    #     diff = pdf1 - pdf2
    #     idx = np.where(np.diff(np.sign(diff)))[0][0]
    #     return x[idx]

    # pdf1 = norm.pdf(x, gmm.means_[0, 0], np.sqrt(gmm.covariances_[0, 0, 0])) * gmm.weights_[0]
    # pdf2 = norm.pdf(x, gmm.means_[1, 0], np.sqrt(gmm.covariances_[1, 0, 0])) * gmm.weights_[1]

    # intersection = find_intersection(x, pdf1, pdf2)
    # print("Intersection point:", intersection)

    # ax.axvline(intersection, color='r', linestyle='--', label='Intersection')
    # ax.xaxis.set_major_locator(plt.MultipleLocator(bin_size))
    ax.set_xlabel(feature)
    ax.set_ylabel('Probability Density')
    ax.set_title('Histogram and Density of Data')
    ax.set_xlim(0, 1000)
    # ax.set_xlim()
    ax.legend()
    plt.savefig('gmmFeaturestest_'+ feature + '.png')

def main(file1):
    features_df = pd.read_csv(file1, sep='\t')
    # chr  start   end     avg_mapq  r0_rate rdv rstd   r0r_rate   depth_mean r0r_depth_mean  depth_cv
    # chrX_df = features_df[features_df['end'] < 2543797]
    chrX_df = features_df
    # X = chrX_df[['avg_mapq', 'depth_cv', 'r0r_rate']]
    X = chrX_df[['avg_mapq', 'depth_mean', 'depth_cv', 'r0r_rate']]
    # scaler = StandardScaler()
    scaler = MinMaxScaler()

    # X_scaled = scaler.fit_transform(X)

    X_scaled = X
    figure1(X)
    # featureDistribution(X, 'depth_mean')
    exit()
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(X_scaled)
    featuresImportance(X_scaled, gmm)
    exit()

    labels = gmm.predict(X_scaled)
    weights = gmm.predict_proba(X_scaled)
    weights1= [sublist[1] for sublist in weights]

    X_df = features_df
    X_df["predicted_cluster"] = labels
    X_df["predicted_weights1"] = weights1
    

    fig = plt.figure(dpi=1200)
    ax = fig.add_subplot(111, projection='3d')

    # Define colors and labels for each cluster
    colors = ['Purple', 'green']
    cluster_labels = ['Cluster 1', 'Cluster 2']

    # Scatter plot for each cluster separately
    for i in range(gmm.n_components):
        cluster_indices = labels == i
        ax.scatter(X_scaled[cluster_indices]['avg_mapq'],
                X_scaled[cluster_indices]['depth_cv'],
                X_scaled[cluster_indices]['r0r_rate'],
                c=colors[i],
                alpha=0.5,
                s=1,
                label=cluster_labels[i])

    # Set axis labels
    ax.set_xlabel('avg_mapq')
    ax.set_ylabel('depth_cv')
    ax.set_zlabel('r0r_rate')
    # ax.set_xlim(54, 60)  # 设置x轴范围
    # ax.set_ylim(50, 60)
    # ax.set_zlim(50, 60)
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)
    ax.tick_params(axis='z', labelsize=5)
    # Add a legend to the plot
    legend = ax.legend(frameon=False, scatterpoints=1, markerscale=5)  # Make the points in the legend solid

    # Set the border around the legend labels to be invisible
    for lh in legend.legend_handles:
        lh.set_edgecolor('none')

    # Set the title of the plot
    ax.set_title('GMM Clustering of Data Points')

    plt.savefig('model_test.png')


    # X_df = pd.DataFrame(X_scaled, columns=['avg_mapq', 'depth_cv', 'r0r_rate'])
    
    # print(len(X_df[X_df["predicted_cluster"] == 0])/len(X_df["predicted_cluster"]))
    zdf = X_df[X_df["predicted_cluster"] == 1]
    # zdf = zdf[zdf['avg_mapq'] > 56]
    # zdf = zdf[zdf['depth_cv'] < 0.2]
    # print(zdf)
    plt.figure(figsize=(9,7),dpi=300)
    sns.scatterplot(data=X_df,
                    x="depth_cv",
                    y="r0r_rate",
                    hue="predicted_cluster",
                    palette=["blue","green"])


    plt.savefig('model_test1.png')


    

    gmm_results_file = '/czlab/xuwen/DFCI/project2/NA12878/analysis/hg38/GMM/gmm_results_file.tsv'
    X_df.to_csv(gmm_results_file, sep='\t', index=False)










if __name__ == '__main__':
    
    features_file = '/czlab/xuwen/DFCI/project2/NA12878/analysis/hg38/GMM/gmmFeatures_final_output.txt'

    main(features_file)
