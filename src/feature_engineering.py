import pandas as pd
import numpy as np

class MalariaFeatureEngineer:
    def __init__(self):
        # Define mild and severe symptoms based on medical literature
        self.mild_symptoms = ['fever', 'cold', 'rigor', 'fatigue', 'headace', 'bitter_tongue', 'vomitting', 'diarrhea']
        self.severe_symptoms = ['Convulsion', 'Anemia', 'jundice', 'cocacola_urine', 'hypoglycemia', 'prostraction', 'hyperpyrexia']
        self.all_symptoms = self.mild_symptoms + self.severe_symptoms

    def fit(self, X, y=None):
        # Nothing to fit, but complies with sklearn transformer API
        return self

    def transform(self, X):
        # Ensure we don't modify the original DataFrame
        X_out = X.copy()
        
        # 1. Age Groups
        X_out['is_child'] = (X_out['age'] <= 12).astype(int)
        X_out['is_elderly'] = (X_out['age'] > 50).astype(int)
        
        # 2. Symptom Counts
        # Ensure only columns that exist are summed
        symptoms_present = [col for col in self.all_symptoms if col in X_out.columns]
        X_out['symptom_count'] = X_out[symptoms_present].sum(axis=1)
        
        severe_present = [col for col in self.severe_symptoms if col in X_out.columns]
        X_out['severe_symptom_count'] = X_out[severe_present].sum(axis=1)
        
        # 3. Interactions
        if 'fever' in X_out.columns and 'rigor' in X_out.columns:
            X_out['fever_rigor'] = (X_out['fever'] * X_out['rigor']).astype(int)
        else:
            X_out['fever_rigor'] = 0

        if 'fever' in X_out.columns and 'headace' in X_out.columns:
            X_out['fever_headache'] = (X_out['fever'] * X_out['headace']).astype(int)
        else:
            X_out['fever_headache'] = 0

        if 'Anemia' in X_out.columns and 'fatigue' in X_out.columns:
            X_out['anemia_fatigue'] = (X_out['Anemia'] * X_out['fatigue']).astype(int)
        else:
            X_out['anemia_fatigue'] = 0

        if 'vomitting' in X_out.columns and 'diarrhea' in X_out.columns:
            X_out['vomiting_diarrhea'] = (X_out['vomitting'] * X_out['diarrhea']).astype(int)
        else:
            X_out['vomiting_diarrhea'] = 0

        # 4. Severe Malaria Score (weighted sum of symptoms: severe weight=2, mild weight=1)
        mild_present = [col for col in self.mild_symptoms if col in X_out.columns]
        X_out['severe_score'] = (X_out[mild_present].sum(axis=1) * 1.0 + 
                                  X_out[severe_present].sum(axis=1) * 2.0)
        
        return X_out

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
