import pandas as pd
import numpy as np


def get_summary(results: list) -> pd.DataFrame:
    """Get summary of the results.

    Args:
        results (list): The results from the model.

    Returns:
        pd.DataFrame: The summary of the results.
    """
    summary = []
    for label in results:
        label = np.array(label)
        values, counts = np.unique(label, return_counts=True)
        summary.append(dict(zip(values.astype(int), counts.astype(int))))
    df = pd.DataFrame(summary).fillna(0).astype(int)
    df.columns = ["bar-scale", "color stamp", "detail label", "north sign"]
    df['bar-scale-present'] = df['bar-scale'].apply(lambda x: True
                                                    if x > 0 else False)
    df['color-stamp-present'] = df['color stamp'].apply(lambda x: True
                                                        if x > 0 else False)
    df['detail-label-present'] = df['detail label'].apply(lambda x: True
                                                          if x > 0 else False)
    df['north-sign-present'] = df['north sign'].apply(lambda x: True
                                                      if x > 0 else False)
    df = df[["bar-scale-present", "bar-scale", "color-stamp-present",
             "color stamp", "detail-label-present", "detail label",
             "north-sign-present", "north sign"]]
    return df
