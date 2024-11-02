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

def FeatureEngineering(df):
    df['Shock'] = df['HR'] / (df['SBP'] + 0.0000001)
    
    # Check if Fever is higher than 38 or lower than 36 at that point or after
    df['Fever'] = df.groupby('patient')['Temp'].transform(
        lambda x: (x < 36).cumsum() + (x > 38).cumsum() > 0
        ).astype(int)
    
    # Check for Tachycardis (HR > 90 BPM)
    df['TCA'] = df.groupby('patient')['HR'].transform(
        lambda x: (x > 90).cumsum() > 0
        ).astype(int)
    
    # Check for Tachypnea (Resp > 20 breaths per minute)
    df['TCP'] = df.groupby('patient')['Resp'].transform(
        lambda x: (x > 20).cumsum() > 0
        ).astype(int)
    
    # Check for Leukocytosis (WBC > 12000) or Leukopenimia (WBC < 4000)
    df['LEU'] = df.groupby('patient')['WBC'].transform(
        lambda x: (x < 4).cumsum() + (x > 12).cumsum() > 0
        ).astype(int)
    
    return df