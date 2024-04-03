import base64
import logging
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from etna.analysis import plot_forecast
from etna.datasets import TSDataset
from etna.metrics import SMAPE, MAE, R2
from etna.models import CatBoostPerSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import (
    DensityOutliersTransform,
    TimeSeriesImputerTransform,
    LinearTrendTransform,
    LagTransform,
    DateFlagsTransform,
    FourierTransform,
    SegmentEncoderTransform,
    MeanTransform,
)
from pandas import DataFrame, read_csv

# Set up logging
logging.basicConfig(level=logging.INFO)

# Configure Streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="CatBoostPerSegment App", page_icon=":rocket:", layout="centered",
                   initial_sidebar_state='expanded')
st.title("CatBoostPerSegment Time Series Forecasting")
st.write(
    "Welcome to the CatBoostPerSegment Model Training and Testing App!",
    "This app allows you to build and evaluate a CatBoostPerSegment model on your data.",
    "You can choose from a variety of ETNA transforms to customize your model's pipeline."
)


@st.cache_resource
def load_data(file) -> DataFrame:
    """
    Load data from CSV file and check required columns exist.
    Return a Pandas DataFrame object.
    """
    data_df = read_csv(file)
    if data_df.empty:
        logging.warning("The provided file is empty, please check the file.")
        st.warning("The uploaded file is empty, please check the file.")
        raise ValueError("Empty file")

    required_cols = ['timestamp', 'segment', 'target']
    missing_cols = [col for col in required_cols if col not in data_df.columns]
    if missing_cols:
        logging.warning(f"The provided file does not have the following columns: {', '.join(missing_cols)}")
        st.warning(f"The uploaded file does not have the following columns: {', '.join(missing_cols)}")
        raise ValueError("Missing required columns")

    return data_df


def create_tsdataset(df: DataFrame) -> TSDataset:
    """
    Create TSDataset object from a Pandas DataFrame.
    """
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df, freq="D")
    return ts


def create_transforms_feature_options(horizon: int) -> Dict[str, object]:
    """
    Create dictionary with available transform options.
    """
    return {
        "LagTransform": LagTransform(in_column="target", lags=list(range(horizon, 122)), out_column="target_lag"),
        "DateFlagsTransform": DateFlagsTransform(week_number_in_month=True, out_column="date_flag"),
        "FourierTransform": FourierTransform(period=360.25, order=6, out_column="fourier"),
        "MeanTransform": MeanTransform(in_column="target", window=12, seasonality=7),
    }


def display_transforms_feature_options(transforms_feature_options: Dict[str, object]) -> List[str]:
    """
    Show list of transform options to user. Return selected transforms.
    """
    return st.multiselect(
        "Select one or more Transforms with feature", list(transforms_feature_options.keys())
    )


def create_pipeline(
        ts: TSDataset,
        transforms_feature_options: Dict[str, object],
        selected_transforms: List[str],
        horizon: int,
        iterations: int,
        learning_rate: int,
        depth: int,
        loss_function: str,
        border_count: int,
        imputation_strategy: str,
        outlier_coef: float,
        seasonality: int
) -> Pipeline | None:
    """
    Create pipeline with user-selected transforms and standard transforms.
    """
    suffix = f"_lag_{horizon}"
    selected_transforms_withfeature = [
        transforms_feature_options[transform_name] for transform_name in selected_transforms
        if transform_name in transforms_feature_options
    ]

    transforms = [
        LagTransform(in_column="target", lags=list(range(horizon, 122)), out_column=f"target{suffix}"),
        MeanTransform(in_column="target", window=12, seasonality=seasonality, out_column=f"mean{suffix}"),
        DensityOutliersTransform(in_column="target", distance_coef=outlier_coef),
        TimeSeriesImputerTransform(in_column="target", strategy=imputation_strategy),
        LinearTrendTransform(in_column="target"),
        SegmentEncoderTransform(),
    ]

    for transform in selected_transforms_withfeature:
        if isinstance(transform, DensityOutliersTransform) and transform not in transforms:
            transforms.append(transform)
        elif transform in transforms:
            st.warning(f"The '{transform}' feature is already being used. Please select a different transform.")
            return None
        else:
            if hasattr(transform, 'fit_transform'):
                ts = transform.fit_transform(ts)
            else:
                logging.warning(f"Warning: transform {transform} does not have a fit_transform method")
                continue
            transforms.append(transform)

    model = CatBoostPerSegmentModel(iterations=iterations, learning_rate=learning_rate, depth=depth,
                                    loss_function=loss_function, border_count=border_count)
    pipeline = Pipeline(transforms=transforms, model=model, horizon=horizon)

    return pipeline


def evaluate_forecast(test_ts: TSDataset, forecast_ts: TSDataset, selected_segment_def) -> Tuple[float, float, float]:
    """
    Calculate evaluation metrics and return results as a Tuple.
    """
    if selected_segment_def is not None:
        smape = SMAPE(mode="per-segment")(y_true=test_ts, y_pred=forecast_ts)[selected_segment_def]
        mae = MAE(mode="per-segment")(y_true=test_ts, y_pred=forecast_ts)[selected_segment_def]
        r2 = R2(mode="per-segment")(y_true=test_ts, y_pred=forecast_ts)[selected_segment_def]
    else:
        smape = SMAPE(mode="macro")(y_true=test_ts, y_pred=forecast_ts)
        mae = MAE(mode="macro")(y_true=test_ts, y_pred=forecast_ts)
        r2 = R2(mode="macro")(y_true=test_ts, y_pred=forecast_ts)
    return smape, mae, r2


def main():
    """
    Main function for Streamlit app.
    """
    # Load data from CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv", key="file_uploader")
    if uploaded_file is not None:
        try:
            data_df = load_data(uploaded_file)
        except ValueError as e:
            st.warning(f"An error occurred while loading the data: {str(e)}")
            return  # Exit the function if there's an error loading the data

        # Create TSDataset and set frequency
        ts = create_tsdataset(data_df)

        data_rep = data_df.pivot(index='timestamp', columns='segment', values='target').reset_index()
        data_rep.index = data_rep.index + 1
        # Display data in webapp
        st.dataframe(data_rep)

        # User input for model parameters
        # Provide tooltips for user input

        iterations = st.sidebar.slider('Number of iterations', 10, 1000, 100, step=10,
                                       help="The number of iterations for the model to run.")
        learning_rate = st.sidebar.slider('Learning rate', 0.01, 1.0, 0.1, step=0.01,
                                          help="The rate at which the model learns from the data.")
        depth = st.sidebar.slider('Depth', 1, 10, 6,
                                  help="The depth of the trees in the model.")
        imputation_strategy = st.sidebar.selectbox('Imputation strategy', ['mean', 'forward_fill'],
                                                   help="The strategy to use for imputing missing values.")
        outlier_coef = st.sidebar.slider('Outlier detection coefficient', 1.0, 5.0, 3.0, step=0.1,
                                         help="The coefficient used for detecting outliers.")
        seasonality = st.sidebar.slider('Seasonality for MeanTransform', 1, 30, 7,
                                        help="The length of the seasonality cycle to use in the MeanTransform.")
        loss_function = st.sidebar.selectbox('Loss function', ['RMSE', 'Quantile', 'MAPE', 'Poisson', 'MAE'],
                                             key="loss_function_select",
                                             help="The loss function to use during model training.")
        border_count = st.sidebar.slider('Border count', 1, 255, 128, key="border_count_slider",
                                         help="The number of splits for numerical features.")

        n_unique_rows = data_df['timestamp'].nunique()
        n_train_samples = st.slider("Select the number of training samples", 50, n_unique_rows,
                                    int(n_unique_rows * 0.8), key="n_train_samples_slider")

        # User input for forecast horizon
        horizon = st.slider("Select the forecast horizon (in days)", min_value=7, max_value=60, value=14, step=1,
                            key="horizon_slider")

        # Choose transforms to include in pipeline
        transforms_feature_options = create_transforms_feature_options(horizon)
        selected_transforms = display_transforms_feature_options(transforms_feature_options)
        if len(selected_transforms) == 0:
            st.warning("Please select at least one transform.")
            return  # Exit the function if no transforms are selected

        # Create pipeline with selected transforms and standard transforms
        pipeline = create_pipeline(ts, transforms_feature_options, selected_transforms, horizon, iterations=iterations,
                                   learning_rate=learning_rate, depth=depth, imputation_strategy=imputation_strategy,
                                   outlier_coef=outlier_coef, seasonality=seasonality, loss_function=loss_function,
                                   border_count=border_count)

        # Split data into testing and training datasets
        train_ts, test_ts = ts.train_test_split(test_size=horizon)

        # User input for segment selection
        segments = [None] + list(set(data_df["segment"]))
        selected_segment = st.selectbox("Select a segment", segments, key="segment_select")
        selected_segment_def = selected_segment
        if selected_segment is not None:
            selected_segment = [selected_segment]

        if st.button('Run forecast'):
            try:
                # Fit pipeline to training data
                pipeline.fit(train_ts)

                # Make predictions on test data
                forecast_ts = pipeline.forecast()

                # Plot forecast
                st.pyplot(
                    plot_forecast(
                        forecast_ts=forecast_ts,
                        test_ts=test_ts,
                        train_ts=train_ts,
                        segments=selected_segment,
                        n_train_samples=n_train_samples,
                    )
                )

                # Calculate evaluation metrics
                smape, mae, r2 = evaluate_forecast(test_ts, forecast_ts, selected_segment_def)
                if selected_segment_def is None:
                    st.subheader("Evaluation Metrics")
                else:
                    segment_name = selected_segment_def.replace("_", " ").capitalize()
                    st.subheader(f"Evaluation Metrics for {segment_name}")
                st.metric(label="SMAPE", value=smape)
                st.metric(label="MAE", value=mae)
                st.metric(label="R-squared", value=r2)

                # Filter forecast for selected segment
                forecast_df = forecast_ts.df
                mean_lag_col = f"mean_lag_{horizon}"

                mean_lag = forecast_df.xs(mean_lag_col, level='feature', axis=1)
                target = forecast_df.xs('target', level='feature', axis=1)

                mean_lag_df = mean_lag.reset_index()
                target_df = target.reset_index()
                if selected_segment_def is not None:
                    # Merge mean_lag_df and target_df
                    results_df = pd.merge(mean_lag_df, target_df, on='timestamp', suffixes=('_mean_lag', '_target'))

                    # Define columns order for the selected segment
                    cols_order = [f'{selected_segment_def}_mean_lag', f'{selected_segment_def}_target']

                    # Reindex the results DataFrame
                    results_df = results_df.reindex(columns=cols_order)

                    # Display the results DataFrame
                    results_df = results_df.rename(
                        columns={results_df.columns[1]: 'Actual', results_df.columns[0]: 'Predicted'})

                else:
                    segments = data_df['segment'].unique()
                    results_df = pd.merge(mean_lag_df, target_df, on='timestamp', suffixes=('_mean_lag', '_target'))

                    # Prepare the new order of columns
                    cols_order = []
                    for segment in segments:
                        cols_order.extend([f'{segment}_mean_lag', f'{segment}_target'])
                    results_df = results_df.reindex(columns=cols_order)

                    # Prepare the new column names
                    new_column_names = {}
                    for segment in segments:
                        new_column_names[f'{segment}_mean_lag'] = f'Predicted {segment}'
                        new_column_names[f'{segment}_target'] = f'Actual {segment}'

                    # Rename the columns
                    results_df = results_df.rename(columns=new_column_names)
                st.subheader("Prediction Results")
                st.dataframe(results_df)

                # Convert forecast to CSV and create download link
                csv = results_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # Some strings <-> bytes conversions necessary here
                filename = f"forecast_{selected_segment_def}.csv" if selected_segment_def else "forecast.csv"
                href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Forecast CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)

            except Exception as e:
                st.warning(f"An error occurred while running the pipeline: {str(e)}")


if __name__ == "__main__":
    main()
