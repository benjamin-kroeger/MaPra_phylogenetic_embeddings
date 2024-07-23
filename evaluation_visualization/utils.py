import seaborn as sns
import pandas as pd
import numpy as np
color_palette = sns.color_palette("colorblind", 200)
color_assignment = {}


def get_color(familiy_name: str) -> str:
    if familiy_name in color_assignment.keys():
        return color_assignment[familiy_name]

    new_color = color_palette[len(color_assignment)]
    color_assignment[familiy_name] = '#{:02x}{:02x}{:02x}'.format(int(new_color[0] * 255), int(new_color[1] * 255), int(new_color[2] * 255))

    return color_assignment[familiy_name]


def _square_to_lower_triangle(dist_square):
    n = len(dist_square)
    dist_lower_triangle = []
    for i in range(n):
        row = []
        for j in range(i + 1):
            row.append(dist_square[i][j])
        dist_lower_triangle.append(row)
    return dist_lower_triangle


def _check_triangularitry(df: pd.DataFrame) -> bool:
    """
    Checks if the df is triangular.
    Args:
        df: The dataframe to check.

    Returns:
        True/False
    """
    return df.where(np.triu(np.ones(df.shape), k=1).astype(bool)).isna().all().all()


def convert_to_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a triangular df into a full dataframe.
    Args:
        df: The triangular df to convert.

    Returns:
        A square df
    """
    df = df.fillna(0)
    full_matrix = df.values + df.values.T
    np.fill_diagonal(full_matrix, 0)

    full_df = pd.DataFrame(full_matrix, columns=df.columns, index=df.index)
    return full_df