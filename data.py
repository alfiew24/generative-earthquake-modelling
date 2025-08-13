from warnings import filterwarnings; filterwarnings('ignore') # To filter warnings for a cleaner log
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde



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
    feature_set = ['Record Sequence Number', 'EQID', 'Earthquake Magnitude', 'Hypocenter Depth (km)', 'ClstD (km)', 'Vs30 (m/s) selected for analysis', 'PGA (g)']
    response_spectrum_features = [c for c in data.columns if c.startswith('T') and c.endswith('S')]

    desired_T_vals = np.concatenate([np.arange(*sec) for sec in T_val_step_list])
    S_a = process_response_spectrum(data[response_spectrum_features], desired_T_vals)

    other_features = data[feature_set]
    
    # Just in case any nulls exist, drop them off before training
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

def undersample(spectrum:pd.DataFrame, data:pd.DataFrame, features:list[str], sample_portion:float=0.5, replace:bool=False, plot:bool=False):
    """
    This function performs undersampling of the datasets based on some provided features in order to make the distribution of samples
    across these features more even. It does this using a KDE-undersampling technique.

    Args:
        spectrum (pandas dataframe): The dataset containing the response spectrum features.
        data (pandas dataframe): The dataset containing all other features including the ones to undersample by.
        features (list): The list of column names to undersample by.
        sample_portion (float): The portion of the dataset that you want to be returned.
        replace (bool): A flag to indicate whether samples should be taken with or without replacement.
        plot (bool): A flag to indicate whether the before and after distributions should be plotted.

    Returns:
        pandas dataframe: The undersampled response spectrum features.
        pandas dataframe: The undersampled other features.
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