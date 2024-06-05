import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


def impute_missing_tabular_data(
    tabular_df: pd.DataFrame, random_state: int = 0, round_num_col: int = 2
) -> pd.DataFrame:
    """Imputes missing tabular values using regression from available values.

    Args:
        tabular_df: Available tabular data extracted from the patients.
        random_state: Random state for the imputer.
        round_num_col: Number of decimal places to round numerical columns to.

    Returns:
        Tabular data with missing values imputed.
    """
    cat_df = tabular_df.select_dtypes(include="category")
    boolean_df = tabular_df.select_dtypes(include=bool)
    num_df = tabular_df.select_dtypes(exclude=["category", bool])

    # Convert categorical variables to one-hot encoding
    tabular_float_df = tabular_df.copy()
    for col_name, col_data in cat_df.items():
        # Manually flag missing values as NaN, since categorical codes mark them as -1
        tabular_float_df[col_name] = col_data.cat.codes.astype(float)
        tabular_float_df.loc[col_data.isna(), col_name] = np.nan

    # Make sure values are floats (leading missing values to be marked as NaN)
    tabular_float_df = tabular_float_df.astype(float)

    # Impute missing values using regression from other variables
    tabular_data = tabular_float_df.to_numpy()
    imp_tabular_data = IterativeImputer(random_state=random_state)
    filled_tabular_data = imp_tabular_data.fit_transform(tabular_data)
    filled_tabular_df = pd.DataFrame(filled_tabular_data, index=tabular_df.index, columns=tabular_df.columns)

    # Convert the one-hot encoded categorical variables back to their original format
    for col_name, col_data in cat_df.items():
        filled_tabular_df[col_name] = pd.Categorical.from_codes(
            filled_tabular_df[col_name].astype("uint8"), categories=col_data.cat.categories
        )
    # Convert the boolean values back to boolean
    filled_tabular_df[boolean_df.columns] = filled_tabular_df[boolean_df.columns].round().astype(bool)
    # Round the numerical values to their original format
    # NOTE: Explicitly round int columns to 0 decimal places to avoid safe casting errors from pandas
    num_dtypes = {col: dtype for col, dtype in num_df.dtypes.items()}
    int_cols = num_df.select_dtypes(include=int).columns
    dec_by_col = {col: 0 if col in int_cols else round_num_col for col in num_dtypes}
    filled_tabular_df = filled_tabular_df.round(dec_by_col).astype(num_dtypes)

    return filled_tabular_df
