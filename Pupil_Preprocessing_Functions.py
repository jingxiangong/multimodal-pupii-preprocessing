import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import mean_squared_error
from scipy.interpolate import Akima1DInterpolator

def process_pupil_data(df, start_time, end_time, min_pupil=2, max_pupil=8):
    """Read pupil measurements from an iMotions export.

    Records start at the first valid paired observation. Within that range,
    keep timestamps with a reading from either eye for later per-eye filtering.
    """
    columns = ['Unnamed: 1', 'Unnamed: 84', 'Unnamed: 85']
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError(f"Missing iMotions columns: {missing}")
    if start_time >= end_time:
        raise ValueError('start_time must be earlier than end_time')

    # iMotions exports have 27 preamble rows followed by the column headings.
    pupil = df[columns].iloc[27:].copy()
    if len(pupil) < 2:
        raise ValueError('No pupil records found after the iMotions preamble')
    pupil.columns = pupil.iloc[0]
    pupil = pupil.iloc[1:].reset_index(drop=True)

    needed = ['Timestamp', 'ET_PupilLeft', 'ET_PupilRight']
    if not all(name in pupil.columns for name in needed):
        raise ValueError('Expected Timestamp and both pupil columns in the export')
    for name in needed:
        pupil[name] = pd.to_numeric(pupil[name], errors='coerce')

    # Keep the original first-valid-pair rule for the start of the recording.
    first_pairs = pupil[
        pupil['ET_PupilLeft'].gt(min_pupil) & pupil['ET_PupilLeft'].lt(max_pupil)
        & pupil['ET_PupilRight'].gt(min_pupil) & pupil['ET_PupilRight'].lt(max_pupil)
    ]
    if first_pairs.empty:
        raise ValueError('No valid paired pupil observation was found')
    first_time = first_pairs.iloc[0]['Timestamp']

    pupil = pupil[
        pupil['Timestamp'].ge(first_time) & pupil['Timestamp'].le(end_time)
    ].copy()
    pupil = pupil.dropna(subset=['Timestamp'])
    pupil = pupil.dropna(subset=['ET_PupilLeft', 'ET_PupilRight'], how='all')
    return pupil.drop_duplicates(subset='Timestamp', keep='first').copy()


def finds_baseline_period(df_filtered, min_pupil=2, max_pupil=8, baseline_period=400):
    """
    Finds the start of the baseline period in the unprocessed dataset.

    Parameters
    ---------
    df_filtered : pandas.DataFrame
        The pupil diameter dataset after remove all invalid data.
    min_pupil : float
        Minimum valid pupil diameter.The default is 2.
    max_pupil : float
        Maximum valid pupil diameter. The default is 8.
    baseline_period : int
        The time period (in milliseconds) used to calculate the baseline pupil diameter.The default is 400 ms.

    Returns
    ------
    start_baseline : float
        The starting timestamp for the baseline period, rounded to 0 decimal places.

    """

    i = 0
    df_baseline = df_filtered[
        (df_filtered['Timestamp'] >= df_filtered['Timestamp'].iloc[0]) &
        (df_filtered['Timestamp'] <= df_filtered['Timestamp'].iloc[0] + baseline_period)
    ]
    # Check if the baseline period contains only valid rows
    while ((df_baseline['ET_PupilLeft'] <= min_pupil).any()|(df_baseline['ET_PupilLeft'] >= max_pupil).any()
            |(df_baseline['ET_PupilRight'] <= min_pupil).any()|(df_baseline['ET_PupilRight'] >= max_pupil).any() |df_baseline.empty):
        i += 1
        df_baseline = df_filtered[
            (df_filtered['Timestamp'] > df_filtered['Timestamp'].iloc[0 + i]) &
            (df_filtered['Timestamp'] <= df_filtered['Timestamp'].iloc[0 + i] + baseline_period)
        ]
    # Get the starting timestamp of the valid baseline range
    start_baseline = df_baseline['Timestamp'].iloc[0]
    # Round to the nearest number for consistency
    start_baseline = start_baseline.round(0)
    return start_baseline

def remove_invalid_pupil(df,min_size=2,max_size=8):
    """
    Remove all the pupil size fall outside between 2 to 8mm

    Parameters
    ----------
    df : pandas.DataFrame
        The pupil diameter data
    min_size : float, optional
        Minimum valid pupil diameter.The default is 2.
    max_size : float, optional
        Maximum valid pupil diameter.The default is 8.

    Returns
    -------
    df_left ： pandas.DataFrame
        DataFrame containing left eye pupil diameter after removing invalid pupil size
    df_right ： pandas.DataFrame
        DataFrame containing right eye pupil diameter after removing invalid pupil size
    """
    df['Timestamp'] = df['Timestamp'].round(0)
    # Filter for valid left pupil data
    df_left = df[['Timestamp', 'ET_PupilLeft']][(df['ET_PupilLeft'] >= min_size) & (df['ET_PupilLeft'] <= max_size)]

    # Filter for valid right pupil data
    df_right = df[['Timestamp', 'ET_PupilRight']][(df['ET_PupilRight'] >= min_size) & (df['ET_PupilRight'] <= max_size)]

    return df_left, df_right

def remove_outliers(df, pupil_column='ET_PupilLeft', time_column='Timestamp', n=20):
    """
    Detects and removes outliers based on Median Absolute Deviation (MAD) method.

    Parameters
    ----------
    df : pandas.DataFrame
        The pupil diameter data
    pupil_column : str
        The name of the pupil diameter column. The default is 'ET_PupilLeft'.
    time_column : str
        The name of the column containing timestamp data. The default is 'Timestamp'.
    n : int
        The scaling factor used to determine the sensitivity of the outlier detection. The default is 20.

    Returns
    -------
    df_cleaned : pandas.DataFrame
        DataFrame after removing the outlier based on MAD method.
    """

    df_pupil = df[pupil_column]
    df_time = df[time_column]
    #Calculate each dilation speed
    d_prime = []

    for i in range(len(df_pupil)):
        if i == 0:
            d_prime.append(abs((df_pupil.iloc[i + 1] - df_pupil.iloc[i]) / (df_time.iloc[i + 1] - df_time.iloc[i])))
        elif i == len(df) - 1:
            d_prime.append(abs((df_pupil.iloc[i] - df_pupil.iloc[i - 1]) / (df_time.iloc[i] - df_time.iloc[i - 1])))
        else:
            d_prev = abs((df_pupil.iloc[i] - df_pupil.iloc[i - 1]) / (df_time.iloc[i] - df_time.iloc[i - 1]))
            d_next = abs((df_pupil.iloc[i + 1] - df_pupil.iloc[i]) / (df_time.iloc[i + 1] - df_time.iloc[i]))
            d_prime.append(max(d_prev, d_next))

    df['d_prime'] = d_prime

    # Calculate the Median Absolute Deviation (MAD)
    median_p = np.median(df['d_prime'])
    mad = []
    for i in range(len(d_prime)):
     mad.append(abs(d_prime[i]-median_p))
    df['mad'] = mad

    # Remove the outlier where it's dilation speed larger than threshold
    median_mad = np.median(df['mad'])
    threshold = median_p + n * median_mad
    df['is_outlier'] = df['d_prime'] > threshold
    df_cleaned = df[df['is_outlier'] == 0]

    df_cleaned = df_cleaned.drop(columns=['d_prime', 'mad', 'is_outlier'])

    return df_cleaned

def remove_outliers_with_moving_average(df, pupil_column='ET_PupilLeft', window_size=23, n=18):
    """
    Detects and removes outliers using a rolling mean and a deviation threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        The pupil diameter data
    pupil_column : str
        The name of the pupil diameter column. The default is 'ET_PupilLeft'.
    window_size : int
        The rolling window size for calculating the moving average. The default is 23.
    n : int
        The scaling factor used to determine the sensitivity of the outlier detection. The default is 18.

    Returns
    -------
    df_cleaned : pandas.DataFrame
        The DataFrame after removing the outlier based on the moving average filtering.
    """

    # Calculate the rolling mean
    df['moving_average'] = df[pupil_column].rolling(window=window_size, center=True).mean()

    # Calculate deviations from the rolling mean
    df['d'] = np.abs(df[pupil_column] - df['moving_average'])

    # Calculate the threshold for outlier detection
    threshold = n * df['d'].median()

    # Identify outliers
    df['is_outlier'] = df['d'] > threshold

    # Mark all rows as not dropped initially
    df['drop'] = False

    # Mark outliers for dropping
    df.loc[df['is_outlier'], 'drop'] = True

    # Remove outliers from the DataFrame
    df_cleaned = df[~df['is_outlier']].reset_index(drop=True)
    df_cleaned = df_cleaned.drop(columns=['moving_average','d', 'is_outlier', 'drop'])
    return df_cleaned

def pchip_interpolation_1000hz(df, pupil_column='ET_PupilLeft', timestamp_column='Timestamp',
                               interp_upsampling_freq=1000, max_gap=250):
    """
    Applies PCHIP interpolation and removes interpolated data where initial dataset contains the gaps exceed 250ms.

    Parameters
    ----------
    df : pandas.DataFrame
        The pupil diameter data
    pupil_column : str
        The name of the pupil diameter column. The default is 'ET_PupilLeft'.
    timestamp_column : str
        The name of the column containing timestamp data. The default is 'Timestamp'.
    interp_upsampling_freq : int
        The frequency after upsampling in Hz. The default: 1000 Hz.
    max_gap : int
        The interpolated data will be remove if the intial datasets contains the gaps larger than max_gap. The default is 250

    Returns
    -------
    df_interpolated : pandas.DataFrame
        DataFrame with PCHIP interpolated pupil diameter after removing the interpolated points where gaps were larger than max_gap.
    """
    # Calculate time gaps from initial dataset
    df['TimeDiff'] = df[timestamp_column].diff()

    df_pupil = df[pupil_column]
    df_time = df[timestamp_column]

    # Generate new upsampled timestamps at 1000 Hz
    t_upsampled = np.arange(df_time.iloc[0], df_time.iloc[-1], 1000 / interp_upsampling_freq)

    # Apply PCHIP interpolation
    pchip_interp = PchipInterpolator(df_time, df_pupil)
    dia_interp = pchip_interp(t_upsampled)

    # Identify time different in initial dataset
    gaps_raw = df['TimeDiff'].iloc[1:]
    gaps_raw = np.append(gaps_raw, gaps_raw.iloc[-1])
    bin_indices = np.digitize(t_upsampled, df[timestamp_column].values)
    # Assign time differences to interpolated points
    gaps_ms = gaps_raw[np.clip(bin_indices - 1, 0, len(gaps_raw) - 1)]

    # Remove interpolated data where gaps were larger than max_gap
    dia_interp[gaps_ms > max_gap] = np.nan

    df_interpolated = pd.DataFrame({
        'Timestamp': t_upsampled.round(0),
        pupil_column: dia_interp
    }).dropna()

    return df_interpolated

def akima_interpolation_1000hz(df, pupil_column='ET_PupilLeft', timestamp_column='Timestamp',
                               interp_upsampling_freq=1000, max_gap=250):
    """
    Applies Akima interpolation and removes interpolated data where initial dataset contains the gaps exceed 250ms.

    Parameters
    ----------
    df : pandas.DataFrame
        The pupil diameter data
    pupil_column : str
        The name of the pupil diameter column. The default is 'ET_PupilLeft'.
    timestamp_column : str
        The name of the column containing timestamp data. The default is 'Timestamp'.
    interp_upsampling_freq : int
        The frequency after upsampling in Hz. The default: 1000 Hz.
    max_gap : int
        The interpolated data will be remove if the intial datasets contains the gaps larger than max_gap. The default is 250

    Returns
    -------
    df_interpolated : pandas.DataFrame
        DataFrame with PCHIP interpolated pupil diameter after removing the interpolated points where gaps were larger than max_gap.
    """
    # Calculate time gaps from initial dataset
    df['TimeDiff'] = df[timestamp_column].diff()
    df_pupil = df[pupil_column]
    df_time = df[timestamp_column]

    # Generate upsampled timestamps at target frequency
    t_upsampled = np.arange(df_time.iloc[0], df_time.iloc[-1], 1000 / interp_upsampling_freq)

    # Apply Akima interpolation
    akima_interp = Akima1DInterpolator(df_time, df_pupil, method='akima')
    dia_interp = akima_interp(t_upsampled)

    # Map time gaps to interpolated points
    gaps_raw = df['TimeDiff'].iloc[1:]
    gaps_raw = np.append(gaps_raw, gaps_raw.iloc[-1])
    bin_indices = np.digitize(t_upsampled, df[timestamp_column].values)
    # Assign time differences to interpolated points
    gaps_ms = gaps_raw[np.clip(bin_indices - 1, 0, len(gaps_raw) - 1)]

    # Remove interpolated data where gaps were larger than max_gap
    dia_interp[gaps_ms > max_gap] = np.nan

    df_interpolated = pd.DataFrame({
        'Timestamp': t_upsampled.round(0),
        pupil_column: dia_interp
    }).dropna()

    return df_interpolated

def makima_interpolation_1000hz(df, pupil_column='ET_PupilLeft', timestamp_column='Timestamp',
                                interp_upsampling_freq=1000, max_gap=250):
    """
    Applies makima interpolation and removes interpolated data where initial dataset contains the gaps exceed 250ms.

    Parameters
    ----------
    df : pandas.DataFrame
        The pupil diameter data
    pupil_column : str
        The name of the pupil diameter column. The default is 'ET_PupilLeft'.
    timestamp_column : str
        The name of the column containing timestamp data. The default is 'Timestamp'.
    interp_upsampling_freq : int
        The frequency after upsampling in Hz. The default: 1000 Hz.
    max_gap : int
        The interpolated data will be remove if the intial datasets contains the gaps larger than max_gap. The default is 250

    Returns
    -------
    df_interpolated : pandas.DataFrame
        DataFrame with PCHIP interpolated pupil diameter after removing the interpolated points where gaps were larger than max_gap.
    """
    # Calculate time gaps from initial dataset
    df['TimeDiff'] = df[timestamp_column].diff()
    df_pupil = df[pupil_column]
    df_time = df[timestamp_column]

    # Generate upsampled timestamps at target frequency
    t_upsampled = np.arange(df_time.iloc[0], df_time.iloc[-1], 1000 / interp_upsampling_freq)

    # Makima interpolation (requires SciPy >= 1.16)
    makima_interp = Akima1DInterpolator(df_time, df_pupil, method='makima')
    dia_interp = makima_interp(t_upsampled)

    # Identify time different in initial dataset
    gaps_raw = df['TimeDiff'].iloc[1:]
    gaps_raw = np.append(gaps_raw, gaps_raw.iloc[-1])
    bin_indices = np.digitize(t_upsampled, df[timestamp_column].values)
    # Assign time differences to interpolated points
    gaps_ms = gaps_raw[np.clip(bin_indices - 1, 0, len(gaps_raw) - 1)]
    # Remove interpolated data where gaps were larger than max_gap
    dia_interp[gaps_ms > max_gap] = np.nan

    df_interpolated = pd.DataFrame({
        'Timestamp': t_upsampled.round(0),
        pupil_column: dia_interp
    }).dropna()

    return df_interpolated


def compute_mean_pupil_size(df_left, df_right):
    """
    Computes the mean pupil diameter after predicting the missing left/right pupil diameters using linear regression.

    Parameters
    ----------
        df_left : pandas.DataFrame
                  The left eye pupil diameter data
        df_right : pandas.DataFrame
                   The left eye pupil diameter data

    Returns
    -------
        df_mean :pandas.DataFrame
                The mean pupil diameter data calculated from df_left and df_right after applying the linear regression to fill the missing diameters.

    """
    # Merge left and right pupil data on Timestamp
    df_mean = pd.merge(df_left, df_right, on='Timestamp', how='outer')

    # Identify missing values
    LwithoutR = df_mean['ET_PupilLeft'].notna() & df_mean['ET_PupilRight'].isna()
    RwithoutL = df_mean['ET_PupilRight'].notna() & df_mean['ET_PupilLeft'].isna()

    # Extract valid data for training regression models
    valid_data = df_mean.dropna(subset=['ET_PupilLeft', 'ET_PupilRight'])

    # Train linear regression models
    model_LtoR = LinearRegression().fit(valid_data[['ET_PupilLeft']], valid_data['ET_PupilRight'])
    model_RtoL = LinearRegression().fit(valid_data[['ET_PupilRight']], valid_data['ET_PupilLeft'])

    if LwithoutR.any():
        df_mean.loc[LwithoutR, 'ET_PupilRight'] = model_LtoR.predict(df_mean.loc[LwithoutR, ['ET_PupilLeft']])

    if RwithoutL.any():
        df_mean.loc[RwithoutL, 'ET_PupilLeft'] = model_RtoL.predict(df_mean.loc[RwithoutL, ['ET_PupilRight']])

    # Compute mean pupil diameter
    df_mean['meanDia'] = df_mean[['ET_PupilLeft', 'ET_PupilRight']].mean(axis=1)

    return df_mean

def evaluate_interpolations(df, pupil_column='ET_PupilLeft', timestamp_column='Timestamp', test_fraction=0.2, random_state=42):
    """
    Computes the performance of three interpolation methods-PCHIP, Akima, and Makima

    Parameters
    ----------
        df : pandas.DataFrame
            The left eye pupil diameter data
        pupil_column : str
            The name of the pupil diameter column. The default is 'ET_PupilLeft'.
        timestamp_column : str
            The name of the timestamp column. The default is 'Timestamp'
        test_fraction : float
            The fraction of data to be randomly selected and held out as the test set. The data points to be removed and predicted via interpolation. The default is 0.2.
        random_state : int
            Random seed for reproducibility of the test/train split. The default is 42

    Returns
    -------
        results : dict
            A dictionary containing the performance of three interpolation. Each key is the method name, and each value is a tuple:

            RMSE : float
                Root Mean Squared Error between the interpolated and actual values at the test points.

            Max Error : float
                Maximum absolute error between the interpolated and actual values at the test points.
    """
    # Sort and sample data
    df = df.sort_values(timestamp_column).reset_index(drop=True)
    x = df[timestamp_column].values
    y = df[pupil_column].values

    # Remove any NaNs from the original data before splitting
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[valid_mask]
    y = y[valid_mask]

    # Randomly select test indices to remove
    rng = np.random.default_rng(random_state)
    n = len(x)
    test_size = int(n * test_fraction)
    test_indices = rng.choice(n, size=test_size, replace=False)
    train_mask = np.ones(n, dtype=bool)
    train_mask[test_indices] = False

    # Prepare train and test sets
    x_train, y_train = x[train_mask], y[train_mask]
    x_test, y_test = x[test_indices], y[test_indices]

    # Fit interpolators on training data
    pchip = PchipInterpolator(x_train, y_train)
    akima = Akima1DInterpolator(x_train, y_train, method='akima')
    makima = Akima1DInterpolator(x_train, y_train, method='makima')

    # Predict at test points
    y_pred_pchip = pchip(x_test)
    y_pred_akima = akima(x_test)
    y_pred_makima = makima(x_test)

    # Compute error metrics, ignoring NaNs
    def get_metrics(y_true, y_pred):
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if np.sum(mask) == 0:
            return np.nan, np.nan
        rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        max_err = np.max(np.abs(y_true[mask] - y_pred[mask]))
        return rmse, max_err

    results = {
        'PCHIP': get_metrics(y_test, y_pred_pchip),
        'Akima': get_metrics(y_test, y_pred_akima),
        'Makima': get_metrics(y_test, y_pred_makima)
    }

    print("Interpolation Method Comparison (RMSE, Max Error):")
    for method, (rmse, max_err) in results.items():
        print(f"{method}: RMSE = {rmse:.4f}, Max Error = {max_err:.4f}")

    return results

def normalize_pupil_dilation(df_mean, start_baseline, start_time, end_time):
    """
    Caculate the baseline pupil diameter and normalizes pupil dilation.

    Parameters
    ----------
    df_mean : pandas.DataFrame
              The eye pupil diameter data contains the mean pupil diameter data
    start_baseline : int
        The starting timestamp for the baseline period
    start_time : int
        The beginning of the test simulation.
    end_time : int
        The end of the test simulation.

    Returns
    -------
    df_mean : pd.DataFrame
                The eye pupil diameter data in the annotation timestamp contains the normalized pupil dilation and the mean normalized dilation
    """

    # Compute the baseline pupil diameter
    df_baseline = df_mean[(df_mean['Timestamp'] >= start_baseline) & (df_mean['Timestamp'] <= start_baseline + 400)]

    baseline = df_baseline['meanDia'].mean()

    # Calculate normalized pupil dilation
    df_mean['baseline'] = baseline
    df_mean['normalized_dilation'] = (df_mean['meanDia'] - df_mean['baseline']) / df_mean['baseline']

    # Remove unnecessary time stamp
    df_mean = df_mean[(df_mean['Timestamp'] >= start_time) & (df_mean['Timestamp'] < end_time)]

    # Compute mean normalized pupil dilation
    mean_normalized_dilation = df_mean['normalized_dilation'].mean()
    df_mean['mean_normalized_dilation'] = mean_normalized_dilation

    return df_mean

def preprocess_integrate_emotions(df, df_pupil, selected_columns, start_time, end_time, target_hz=30):
    """Aggregate pupil and facial-expression data into the same time bins.

    Pupil values are averaged within a bin. For facial-expression values,
    the first non-missing entry is retained. Unmatched bins remain in the output.
    """
    if start_time >= end_time:
        raise ValueError('start_time must be earlier than end_time')
    if target_hz <= 0 or target_hz > 1000:
        raise ValueError('target_hz must be between 0 and 1000')
    missing = [name for name in selected_columns if name not in df.columns]
    if missing:
        raise ValueError(f"Missing selected iMotions columns: {missing}")
    if 'Timestamp' not in df_pupil.columns:
        raise ValueError('df_pupil must contain a Timestamp column')

    bin_size_ms = int(1000 / target_hz)
    emotion = df[selected_columns].iloc[27:].copy()
    if len(emotion) < 2:
        raise ValueError('No facial-expression records found after the iMotions preamble')
    emotion.columns = emotion.iloc[0]
    emotion = emotion.iloc[1:].reset_index(drop=True)
    if 'Timestamp' not in emotion.columns:
        raise ValueError('Selected facial-expression columns must contain Timestamp')

    emotion['Timestamp'] = pd.to_numeric(emotion['Timestamp'], errors='coerce')
    for name in emotion.columns:
        if name != 'Timestamp':
            # Preserve text-valued columns instead of forcing them to missing.
            try:
                emotion[name] = pd.to_numeric(emotion[name], errors='raise')
            except (TypeError, ValueError):
                pass
    emotion = emotion.dropna(subset=['Timestamp'])
    emotion = emotion[
        emotion['Timestamp'].ge(start_time) & emotion['Timestamp'].lt(end_time)
    ].copy()
    emotion_columns = [name for name in emotion.columns if name != 'Timestamp']
    if emotion_columns:
        emotion = emotion.dropna(subset=emotion_columns, how='all')

    pupil = df_pupil.copy()
    pupil['Timestamp'] = pd.to_numeric(pupil['Timestamp'], errors='coerce')
    pupil = pupil.dropna(subset=['Timestamp'])
    pupil = pupil[
        pupil['Timestamp'].ge(start_time) & pupil['Timestamp'].lt(end_time)
    ].copy()
    pupil_columns = [name for name in pupil.columns if name != 'Timestamp']
    for name in pupil_columns:
        pupil[name] = pd.to_numeric(pupil[name], errors='coerce')

    # Both sources use the same bin key; original timestamps can differ inside a bin.
    pupil['bin'] = ((pupil['Timestamp'] - start_time) // bin_size_ms).astype(int)
    emotion['bin'] = ((emotion['Timestamp'] - start_time) // bin_size_ms).astype(int)
    pupil_bins = pupil.groupby('bin', as_index=False)[pupil_columns].mean()
    emotion_bins = emotion.drop(columns=['Timestamp']).groupby('bin', as_index=False).first()

    result = pd.merge(pupil_bins, emotion_bins, on='bin', how='outer', validate='one_to_one')
    result['Timestamp'] = start_time + result['bin'] * bin_size_ms
    result = result.sort_values('Timestamp').reset_index(drop=True)
    columns = ['Timestamp'] + [name for name in result.columns if name not in ('Timestamp', 'bin')]
    return result[columns]

def plot_pupil_diameter(df,start_time,end_time):
    """
    Visualizes pupil diameter data over time

    Parameters
    ---------
    start_time : int
        The beginning of the test simulation.
    end_time : int
        The end of the test simulation.
    df : pandas.DataFrame
        The processed multimodal dataset containing ET_PupilLeft, ET_PupilRight and meanDia.

    Returns
    -------
    a visual representation of pupil diameter data over time
    """

    # Filter for the timestamp range 2000 to 3000 ms
    filtered_df = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] <= end_time)]

    # Create a new figure with a specified size (width = 12inches, height = 6inches)
    plt.figure(figsize=(12, 6))

    # Plot the 'ET_PupilLeft' column against 'Timestamp' in red
    plt.plot(filtered_df['Timestamp'], filtered_df['ET_PupilLeft'], label='ET_PupilLeft', color='red', marker='o', linestyle='None', markersize=1)

    # Plot the 'ET_PupilRight' column against 'Timestamp' in blue
    plt.plot(filtered_df['Timestamp'], filtered_df['ET_PupilRight'], label='ET_PupilRight', color='blue', marker='o', linestyle='None', markersize=1)

    # Plot the 'meanDia' column against 'Timestamp' in green
    plt.plot(filtered_df['Timestamp'], filtered_df['meanDia'], label='Mean Pupil Diameter', color='green', marker='o', linestyle='None', markersize=1)

    # Set the label for the x-axis
    plt.xlabel('Timestamp (ms)')

    # Set the label for the y-axis
    plt.ylabel('Pupil Diameter (mm)')

    # Set the plot title
    plt.title('Processed Pupil Diameter')
    # Show the legend
    plt.legend()

    plt.tight_layout()

    # Display the plot window
    plt.show()

def plot_left_pupil(df, pupil_column='ET_PupilLeft', timestamp_column='Timestamp',
                    start_time=None, end_time=None, y_min=None, y_max=None):
    """
    Plots the left pupil diameter over time, with optional axis limits.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing pupil data.
    pupil_column : str
        Column name for left pupil diameter (default 'ET_PupilLeft').
    timestamp_column : str
        Column name for timestamps (default 'Timestamp').
    start_time : int or float, optional
        Lower x-axis limit (timestamp).
    end_time : int or float, optional
        Upper x-axis limit (timestamp).
    y_min : float, optional
        Lower y-axis limit.
    y_max : float, optional
        Upper y-axis limit.
    """
    plt.figure(figsize=(6, 6))
    plt.plot(df[timestamp_column], df[pupil_column], color='red', marker='o', linestyle='None', markersize=1, label=pupil_column)
    plt.xlabel('Timestamp (ms)')
    plt.ylabel('Pupil Diameter (mm)')
    plt.title('Left Pupil Diameter Over Time')
    plt.legend()
    plt.tight_layout()

    # Set x-axis limits if provided
    if start_time is not None and end_time is not None:
        plt.xlim(start_time, end_time)  # [3][4][5]
    # Set y-axis limits if provided
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)  # [2][3][4][5]

    plt.show()



