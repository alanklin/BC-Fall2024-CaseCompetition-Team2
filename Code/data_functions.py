def subsequent(df):

    # This function will take the raw submission file (two columns, ID and SepsisLabel) and it makes sure that if a Sepsis label of 1 appears for a patient, that all subsequent
    # rows for the same patient will also predict 1. 

    df[['patient_id', 'time']] = df['ID'].str.split('_', expand=True)
    df['time'] = df['time'].astype(int)


    # Apply the rule: if there's a 1 at any time point, set all later values to 1 for that patient
    df['SepsisLabel'] = df.groupby('patient_id')['SepsisLabel'].transform(lambda x: x.cumsum().clip(upper=1))

    # Combine patient ID and time back into a single column if needed
    df['patient_time'] = df['patient_id'] + '_' + df['time'].astype(str)
    df = df[['ID','SepsisLabel']]

    return df