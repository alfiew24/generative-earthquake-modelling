import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from cycler import cycler
import seaborn as sns

palette = sns.color_palette("Set2")
plt.rcParams.update({
    'axes.prop_cycle': cycler('color', plt.get_cmap('Set2').colors),
    'image.cmap': 'terrain'
})


# This file contains functions to help with reading and pre-processing the dataset.


def process_response_spectrum(spectrum_df:pd.DataFrame, desired_T:np.ndarray):
    """
    This function does the pre-processing to the response spectrum, resampling through interpolation and natural log.

    Args:
        spectrum_df (pandas dataframe): A dataframe which solely contains the response spectrum fields of the data.
        desired_T (numpy array): An array containing the list of desired response spectrum period values.
    
    Returns:
        pandas dataframe: An adjusted dataframe with the interpolated and natural log response spectrum (headers are adjusted too).
    """
    T = np.array([float(c.replace('T', '').replace('S', '')) for c in spectrum_df.columns.tolist()])
    spectrum = spectrum_df.interpolate(axis=1, limit_direction='both').to_numpy() # Fills any null values with interpolation

    interpolator = interp1d(T, spectrum, axis=1, kind='linear', bounds_error=True)
    spectrum = interpolator(desired_T)
    spectrum = np.log(spectrum)

    return pd.DataFrame(spectrum, columns=[f'T{t:.2f}S' for t in np.nditer(desired_T)])



def get_data_for_model(filepath:str, T_val_step_list:list=[[0.1, 1, 0.1], [1, 5, 1.0]]):
    """
    This function reads in the dataset and pre-processes it ready for the VAE model.

    Args:
        filepath (string): Path to the .xlsx file containing the data.
        T-val_step_list (list): A list of lists for the period resampling of the response spectrum, [start, end, step].

    Returns:
        pandas dataframe: The acceleration response spectrum features to be fed to the model.
        pandas dataframe: The other features to be used in the regularisation loss term of the model.
    """
    data = pd.read_excel(filepath)
    data.replace(-999.0, np.nan, inplace=True) # I assume the -999 values are null placeholders?

    # Defining the features of interest
    feature_set = ['EQID', 'Earthquake Magnitude', 'Hypocenter Depth (km)', 'EpiD (km)', 'HypD (km)', 'ClstD (km)', 'Vs30 (m/s) selected for analysis', 'PGA (g)', 'Dip (deg)', 'T-plunge (deg)',
                   'Strike (deg)', 'P-trend (deg)', 'T-trend (deg)', 'Depth to Top Of Fault Rupture Model', 'Fault Rupture Length for Calculation of Ry (km)', 'Fault Rupture Width (km)', 'Fault Rupture Area (km^2)', 'Mechanism Based on Rake Angle', 'Rake Angle (deg)']
    response_spectrum_features = [c for c in data.columns if c.startswith('T') and c.endswith('S') and float(c[1:-1]) <= 5.0]    

    desired_T_vals = np.concatenate([np.arange(*sec[:2], sec[2]) for sec in T_val_step_list])
    S_a = process_response_spectrum(data[response_spectrum_features], desired_T_vals)

    other_features = data[feature_set]

    # Imputating the null values with the average based on earthquake magnitude
    for feature in other_features.columns.tolist():
        other_features[feature] = other_features.groupby('Earthquake Magnitude')[feature].transform(lambda x: x.fillna(x.mean()))
    
    # Just in case any nulls remain, drop them off before training
    S_a = S_a[other_features.notna().all(axis=1) & S_a.notna().all(axis=1)]
    other_features = other_features[other_features.notna().all(axis=1) & S_a.notna().all(axis=1)]

    # Moving PGA from other_features to spectrum and transforming it
    S_a = pd.concat([other_features['PGA (g)'].apply(np.log), S_a], axis=1)
    other_features.drop(columns=['PGA (g)'])

    # Adding number of samples as a column for filtering
    data_g = other_features.groupby('EQID').count()
    data_g = data_g.rename(columns={'Earthquake Magnitude':'Num Samples'})[['Num Samples']]
    other_features = other_features.join(data_g, on='EQID', how='left')

    print(f'Response spectrum size : {S_a.shape}\tOther features size : {other_features.shape}')
    return S_a, other_features



def resample(spectrum:pd.DataFrame, data:pd.DataFrame, features:list[str], sample_portion:float=0.5, replace:bool=False, plot:bool=False):
    """
    This function performs resampling of the datasets based on some provided features in order to make the distribution of samples
    across these features more even. It does this using a KDE-resampling technique.

    Args:
        spectrum (pandas dataframe): The dataset containing the response spectrum features.
        data (pandas dataframe): The dataset containing all other features including the ones to resample by.
        features (list): The list of column names to resample by.
        sample_portion (float): The portion of the dataset that you want to be returned.
        replace (bool): A flag to indicate whether samples should be taken with or without replacement.
        plot (bool): A flag to indicate whether the before and after distributions should be plotted.

    Returns:
        pandas dataframe: The resampled response spectrum features.
        pandas dataframe: The resampled other features.
    """
    data_grouped = data[['EQID', 'Num Samples', *features]].groupby('EQID').mean()
    data_ = data_grouped[features].values.T  # Transpose for KDE input

    kde = gaussian_kde(data[features].values.T, bw_method='silverman')
    density = (kde(data_)**1.0) * data_grouped['Num Samples']
    weights = 1 / density
    weights /= weights.sum()  # Normalise to sum to 1

    # Sample from the dataset using computed weights
    sample_eqids = data_grouped.sample(n=len(data_grouped), weights=weights, replace=replace).index
    data_new = data[data['EQID'].isin(sample_eqids.to_numpy(int)[:int(sample_portion)])]

    n = 0
    while len(data_new)/len(data) < sample_portion:
        data_new = pd.concat([data_new, data[data['EQID'] == sample_eqids[int(sample_portion)+n]]])
        n += 1

    spectrum_new = spectrum.loc[data_new.index]
    
    # Plotting if required
    if plot:

        def histograms(ax_, feat, orientation):
            # First histogram: full data (density=True)
            counts, bins, _ = ax_.hist(data[feat], bins=20, density=True, edgecolor='black', label='Full', orientation=orientation)

            # Second histogram: resampled data (scale to match density scale of first)
            counts_new, _ = np.histogram(data_new[feat], bins=bins)
            bin_widths = np.diff(bins)
            density_new = sample_portion * counts_new / (np.sum(counts_new) * bin_widths)  # proper density scale

            # Plot second histogram manually
            if orientation == 'vertical':
                ax_.bar([], [])
                ax_.bar(bins[:-1], density_new, width=bin_widths, align='edge', edgecolor='black', label='Undersampled')
            else:
                ax_.barh([], [])
                ax_.barh(bins[:-1], density_new, height=bin_widths, align='edge', edgecolor='black', label='Undersampled')

        #fig, ax = plt.subplots(2, 5, gridspec_kw={'width_ratios': [3, 1, 1, 3, 1], 'height_ratios':[1, 3]}, figsize=(13.5, 6))
        #fig.subplots_adjust(wspace=0, hspace=0)
        #ax[0][1].axis('off')
        #ax[1][0].scatter(data[features[0]], data[features[1]], s=20, edgecolor='black', linewidth=0.5)
        #ax[1][0].scatter(data_new[features[0]], data_new[features[1]], s=20, edgecolor='black', linewidth=0.5)
        ##ax[0][0].hist(data[features[0]], edgecolor='black', label='Full', bins=20, density=True)
        ##ax[0][0].hist(data_new[features[0]], edgecolor='black', label='Resampled', bins=20, density=True)
        ##ax[1][1].hist(data[features[1]], edgecolor='black', orientation='horizontal', bins=20, density=True)
        ##ax[1][1].hist(data_new[features[1]], edgecolor='black', orientation='horizontal', bins=20, density=True)
        #histograms(ax[0][0], features[0], 'vertical'); histograms(ax[1][1], features[1], 'horizontal')
        #ax[0][0].spines['top'].set_visible(False); ax[0][0].spines['right'].set_visible(False); ax[0][0].spines['left'].set_visible(False)
        #ax[1][1].spines['top'].set_visible(False); ax[1][1].spines['right'].set_visible(False); ax[1][1].spines['bottom'].set_visible(False)
        #ax[1][0].set_xlabel(r'$M$', fontsize=14)
        #ax[1][0].set_ylabel(r'$Z_{hyp}$ (km)', fontsize=14)
        #ax[0][0].set_xticks([]); ax[0][0].set_yticks([])
        #ax[1][1].set_xticks([]); ax[1][1].set_yticks([])
        #ax[1][0].set_yticks([0, 4, 8, 12, 16, 20])
        #ax[1][0].text(0.5, -0.2, '(a)', transform=ax[1][0].transAxes, ha='center', va='top', fontsize=13, fontweight='bold')
        #ax[1][0].tick_params(axis='both', labelsize=13)

        #ax[0][2].axis('off'); ax[1][2].axis('off')

        #ax[0][4].axis('off')
        #ax[1][3].scatter(data[features[2]], data[features[3]], s=7, edgecolor='black', linewidth=0.2)
        #ax[1][3].scatter(data_new[features[2]], data_new[features[3]], s=7, edgecolor='black', linewidth=0.2)
        ##ax[0][3].hist(data[features[2]], edgecolor='black', label='Full', bins=20, density=True)
        ##ax[0][3].hist(data_new[features[2]], edgecolor='black', label='Resampled', bins=20, density=True)
        ##ax[1][4].hist(data[features[3]], edgecolor='black', orientation='horizontal', bins=20, density=True)
        ##ax[1][4].hist(data_new[features[3]], edgecolor='black', orientation='horizontal', bins=20, density=True)
        #histograms(ax[0][3], features[2], 'vertical'); histograms(ax[1][4], features[3], 'horizontal')
        #ax[0][3].spines['top'].set_visible(False); ax[0][3].spines['right'].set_visible(False); ax[0][3].spines['left'].set_visible(False)
        #ax[1][4].spines['top'].set_visible(False); ax[1][4].spines['right'].set_visible(False); ax[1][4].spines['bottom'].set_visible(False)
        #ax[1][3].set_xlabel(r'$R_{rup}$ (km)', fontsize=14)
        #ax[1][3].set_ylabel(r'$V_{s30}$ (ms$^{-1}$)', fontsize=14)
        #ax[0][3].set_xticks([]); ax[0][3].set_yticks([])
        #ax[1][4].set_xticks([]); ax[1][4].set_yticks([])
        #ax[1][3].set_xticks([0, 75, 150, 225, 300])
        #ax[1][3].set_yticks([100, 225, 350, 475, 600, 725])
        #ax[1][3].text(0.5, -0.2, '(b)', transform=ax[1][3].transAxes, ha='center', va='top', fontsize=13, fontweight='bold')
        #ax[0][3].legend(loc='upper center', bbox_to_anchor=(1.17, 0.8), fontsize=12)
        #ax[1][3].tick_params(axis='both', labelsize=13)
        
        # Add KDE to histograms
        #ax[0][0].set_prop_cycle(color=palette); ax[1][1].set_prop_cycle(color=palette)
        #ax[0][3].set_prop_cycle(color=palette); ax[1][4].set_prop_cycle(color=palette)
        #sns.kdeplot(data[features[0]], ax=ax[0][0], linewidth=1.2, zorder=5, alpha=0.5)
        #sns.kdeplot(data_new[features[0]], ax=ax[0][0], linewidth=1.2, zorder=5, alpha=0.5)
        #sns.kdeplot(data[features[1]], ax=ax[1][1], linewidth=1.2, vertical=True, zorder=5, alpha=0.5)
        #sns.kdeplot(data_new[features[1]], ax=ax[1][1], linewidth=1.2, vertical=True, zorder=5, alpha=0.5)
        #sns.kdeplot(data[features[2]], ax=ax[0][3], linewidth=1.2, zorder=5, alpha=0.5)
        #sns.kdeplot(data_new[features[2]], ax=ax[0][3], linewidth=1.2, zorder=5, alpha=0.5)
        #sns.kdeplot(data[features[3]], ax=ax[1][4], linewidth=1.2, vertical=True, zorder=5, alpha=0.5)
        #sns.kdeplot(data_new[features[3]], ax=ax[1][4], linewidth=1.2, vertical=True, zorder=5, alpha=0.5)

        #sns.kdeplot(data[features[0]], ax=ax[0][0], color='black', linewidth=2.2, zorder=4, alpha=0.5)
        #sns.kdeplot(data_new[features[0]], ax=ax[0][0], color='black', linewidth=2.2, zorder=4, alpha=0.5)
        #sns.kdeplot(data[features[1]], ax=ax[1][1], color='black', linewidth=2.2, vertical=True, zorder=4, alpha=0.5)
        #sns.kdeplot(data_new[features[1]], ax=ax[1][1], color='black', linewidth=2.2, vertical=True, zorder=4, alpha=0.5)
        #sns.kdeplot(data[features[2]], ax=ax[0][3], color='black', linewidth=2.2, zorder=4, alpha=0.5)
        #sns.kdeplot(data_new[features[2]], ax=ax[0][3], color='black', linewidth=2.2, zorder=4, alpha=0.5)
        #sns.kdeplot(data[features[3]], ax=ax[1][4], color='black', linewidth=2.2, vertical=True, zorder=4, alpha=0.5)
        #sns.kdeplot(data_new[features[3]], ax=ax[1][4], color='black', linewidth=2.2, vertical=True, zorder=4, alpha=0.5)

        fig, ax = plt.subplots(5, 2, gridspec_kw={'height_ratios': [1, 3, 1, 1, 3], 'width_ratios':[3, 1]}, figsize=(6, 13.5))
        fig.subplots_adjust(wspace=0, hspace=0)
        ax[0][1].axis('off')
        ax[1][0].scatter(data[features[0]], data[features[1]], s=20, edgecolor='black', linewidth=0.5)
        ax[1][0].scatter(data_new[features[0]], data_new[features[1]], s=20, edgecolor='black', linewidth=0.5)
        histograms(ax[0][0], features[0], 'vertical'); histograms(ax[1][1], features[1], 'horizontal')
        ax[0][0].spines['top'].set_visible(False); ax[0][0].spines['right'].set_visible(False); ax[0][0].spines['left'].set_visible(False)
        ax[1][1].spines['top'].set_visible(False); ax[1][1].spines['right'].set_visible(False); ax[1][1].spines['bottom'].set_visible(False)
        ax[1][0].set_xlabel(r'$M$', fontsize=14)
        ax[1][0].set_ylabel(r'$Z_{hyp}$ (km)', fontsize=14)
        ax[0][0].set_xticks([]); ax[0][0].set_yticks([])
        ax[1][1].set_xticks([]); ax[1][1].set_yticks([])
        ax[1][0].set_yticks([0, 4, 8, 12, 16, 20])
        ax[1][0].text(0.5, -0.2, '(a)', transform=ax[1][0].transAxes, ha='center', va='top', fontsize=13, fontweight='bold')
        ax[1][0].tick_params(axis='both', labelsize=13)

        ax[2][0].axis('off'); ax[2][1].axis('off')

        ax[3][1].axis('off')
        ax[4][0].scatter(data[features[2]], data[features[3]], s=7, edgecolor='black', linewidth=0.2)
        ax[4][0].scatter(data_new[features[2]], data_new[features[3]], s=7, edgecolor='black', linewidth=0.2)
        histograms(ax[3][0], features[2], 'vertical'); histograms(ax[4][1], features[3], 'horizontal')
        ax[3][0].spines['top'].set_visible(False); ax[3][0].spines['right'].set_visible(False); ax[3][0].spines['left'].set_visible(False)
        ax[4][1].spines['top'].set_visible(False); ax[4][1].spines['right'].set_visible(False); ax[4][1].spines['bottom'].set_visible(False)
        ax[4][0].set_xlabel(r'$R_{rup}$ (km)', fontsize=14)
        ax[4][0].set_ylabel(r'$V_{s30}$ (ms$^{-1}$)', fontsize=14)
        ax[3][0].set_xticks([]); ax[3][0].set_yticks([])
        ax[4][1].set_xticks([]); ax[4][1].set_yticks([])
        ax[4][0].set_xticks([0, 75, 150, 225, 300])
        ax[4][0].set_yticks([100, 225, 350, 475, 600, 725])
        ax[4][0].text(0.5, -0.2, '(b)', transform=ax[4][0].transAxes, ha='center', va='top', fontsize=13, fontweight='bold')
        ax[3][0].legend(loc='center', bbox_to_anchor=(1.17, 1.0), fontsize=12)
        ax[4][0].tick_params(axis='both', labelsize=13)
        
        # Add KDE to histograms
        ax[0][0].set_prop_cycle(color=palette); ax[1][1].set_prop_cycle(color=palette)
        ax[3][0].set_prop_cycle(color=palette); ax[4][1].set_prop_cycle(color=palette)
        sns.kdeplot(data[features[0]], ax=ax[0][0], linewidth=1.2, zorder=5, alpha=0.5)
        sns.kdeplot(data_new[features[0]], ax=ax[0][0], linewidth=1.2, zorder=5, alpha=0.5)
        sns.kdeplot(data[features[1]], ax=ax[1][1], linewidth=1.2, vertical=True, zorder=5, alpha=0.5)
        sns.kdeplot(data_new[features[1]], ax=ax[1][1], linewidth=1.2, vertical=True, zorder=5, alpha=0.5)
        sns.kdeplot(data[features[2]], ax=ax[3][0], linewidth=1.2, zorder=5, alpha=0.5)
        sns.kdeplot(data_new[features[2]], ax=ax[3][0], linewidth=1.2, zorder=5, alpha=0.5)
        sns.kdeplot(data[features[3]], ax=ax[4][1], linewidth=1.2, vertical=True, zorder=5, alpha=0.5)
        sns.kdeplot(data_new[features[3]], ax=ax[4][1], linewidth=1.2, vertical=True, zorder=5, alpha=0.5)

        sns.kdeplot(data[features[0]], ax=ax[0][0], color='black', linewidth=2.2, zorder=4, alpha=0.5)
        sns.kdeplot(data_new[features[0]], ax=ax[0][0], color='black', linewidth=2.2, zorder=4, alpha=0.5)
        sns.kdeplot(data[features[1]], ax=ax[1][1], color='black', linewidth=2.2, vertical=True, zorder=4, alpha=0.5)
        sns.kdeplot(data_new[features[1]], ax=ax[1][1], color='black', linewidth=2.2, vertical=True, zorder=4, alpha=0.5)
        sns.kdeplot(data[features[2]], ax=ax[3][0], color='black', linewidth=2.2, zorder=4, alpha=0.5)
        sns.kdeplot(data_new[features[2]], ax=ax[3][0], color='black', linewidth=2.2, zorder=4, alpha=0.5)
        sns.kdeplot(data[features[3]], ax=ax[4][1], color='black', linewidth=2.2, vertical=True, zorder=4, alpha=0.5)
        sns.kdeplot(data_new[features[3]], ax=ax[4][1], color='black', linewidth=2.2, vertical=True, zorder=4, alpha=0.5)

        plt.savefig('COMM514 - Research Project/Final Figures/Undersampling.png', bbox_inches='tight', dpi=400)

    return spectrum_new, data_new



def train_test_split(spectrum:pd.DataFrame, data:pd.DataFrame, test_portion:float=0.3, high_mag_testing:bool=True):
    """
    Splits the data into training and testing subsets, ensuring samples from the same event remain together.
    """
    group_sizes = data.groupby('EQID').size().reset_index(name='size')
    shuffled = group_sizes.sample(frac=1)
    test_size = int(test_portion * len(data))

    test_eqids, test_count = [], 0

    if 'Earthquake Magnitude' in data.columns and high_mag_testing:
        data_high_mag = data[data['Earthquake Magnitude'] == data['Earthquake Magnitude'].max()]
        test_eqids.extend(data_high_mag['EQID'].unique().tolist())
        test_count += len(data_high_mag)
        shuffled = shuffled[~shuffled['EQID'].isin(test_eqids)]

    for _, row in shuffled.iterrows():
        if test_count + row['size'] > test_size: break
        test_eqids.append(row['EQID'])
        test_count += row['size']

    train_data, test_data = data[~data['EQID'].isin(test_eqids)], data[data['EQID'].isin(test_eqids)]
    train_spectrum, test_spectrum = spectrum.loc[train_data.index], spectrum.loc[test_data.index]
    return train_spectrum, test_spectrum, train_data, test_data



if __name__ == '__main__':
    spectrum, data = get_data_for_model('COMM514 - Research Project/Updated_NGA_West2_Flatfile_RotD50_d005_public_version.xlsx',
                                        #[[0.01, 0.5, 0.01], [0.5, 1.0, 0.05], [1.0, 2.0, 0.1], [2.0, 5.5, 0.5]])
                                        [[0.02, 1.5, 0.02], [1.5, 2.5, 0.1], [2.5, 5.5, 0.5]]) # 91
                                        #[[0.02, 1.0, 0.02], [1.0, 2.0, 0.1], [2.0, 5.5, 0.5]]) # 67
    
    filt = (data['Hypocenter Depth (km)'] < 21) & (data['ClstD (km)'] < 300) & (data['Vs30 (m/s) selected for analysis'] < 750)
    #filt = (spectrum['PGA (g)'] > np.log(0.01)) & (data['ClstD (km)'] < 300) & (data['Vs30 (m/s) selected for analysis'] < 750) & (data['Hypocenter Depth (km)'] < 21) & (data['Num Samples'] > 2)

    print(spectrum.shape, len(data['EQID'].unique()))

    spectrum, data = spectrum[filt], data[filt]
    print(spectrum.shape, len(data['EQID'].unique()))
    np.random.seed(12)
    resample(spectrum, data, ['Earthquake Magnitude', 'Hypocenter Depth (km)', 'ClstD (km)', 'Vs30 (m/s) selected for analysis'], 0.5, plot=True)
    
    print(spectrum.shape, len(data['EQID'].unique()))
    #spectrum.to_csv('COMM514 - Research Project/spectrum_010725_2.csv', index=False)
    #data.to_csv('COMM514 - Research Project/data_010725_2.csv', index=False)
