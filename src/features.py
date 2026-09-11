def add_coefficient_of_variation(df):
    df = df.copy()
    df["cv_profile"] = df["std_profile"] / df["mean_profile"]
    return df