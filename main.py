import os
import logging
import subprocess
import threading
import glob
from datetime import datetime
import pandas as pd
import re
import requests
from urllib.parse import urlparse, quote
import platform
import ctypes
import locale
import tempfile
from io import StringIO
from werkzeug.utils import secure_filename
from pubchempy import get_compounds
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import uuid
import time
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import numpy as np
import json
import traceback
import base64
from io import BytesIO
import warnings
import math
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from flask import Flask, render_template, request, jsonify, Response, send_file, redirect, url_for, flash

warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
logging.basicConfig(level=logging.DEBUG)

rowcount = 0
progress_data = {}
smiles_cache = {}

DATASET_LOG_DIR2 = 'static/datasets'
OUTPUTS_DIR = 'static/datasets'
CODE_LOG_DIR = 'code'
IMAGES_LOG_DIR = 'static/images'
RESULTS_DIR = 'static/outputs' 
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'static/datasets')
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
TRAINED_MODELS_DIR = 'static/models'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(CODE_LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IMAGES_LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)

STD_OUTPUT_HANDLE = -11
ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004

class WindowsConsole:
    def __init__(self):
        self.kernel32 = ctypes.windll.kernel32
        self.setup_console()
    
    def setup_console(self):
        handle = self.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        mode = ctypes.c_uint32()
        self.kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        mode.value |= ENABLE_VIRTUAL_TERMINAL_PROCESSING
        self.kernel32.SetConsoleMode(handle, mode)

if platform.system() == "Windows":
    windows_console = WindowsConsole()
    locale.setlocale(locale.LC_ALL, 'turkish')

class SuzukiReactionPredictor:
    def __init__(self):
        self.model = None
        self.current_model_name = 'Hist Gradient Boosting'
        self.df = None
        self.original_df = None
        self.label_encoders = {}
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, min_samples_split=5, random_state=42),
            'Hist Gradient Boosting': HistGradientBoostingRegressor(max_iter=200, max_depth=5, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1),
            'LightGBM': LGBMRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1),
            'SVR': Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))]),
            'Neural Network': Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(hidden_layer_sizes=(100,50), activation='relu', solver='adam', random_state=42, max_iter=500))])
        }
        
        self.chemical_intuition = {
            'optimal_temp': 90,
            'temp_range': 40,
            'optimal_time': 12,
            'time_range': 8,
            'catalyst_k_m': 0.005,
            'catalyst_v_max': 15,
            'min_catalyst': 0.001,
            'max_catalyst': 0.1,
            'steric_threshold': 0.4,
            'steric_penalty': 6,
            'quality_bonus': 4,
            'max_yield': 100,
            'min_yield': 10,
            'temp_degradation_threshold': 150,
            'temp_too_low_threshold': 40,
            'catalyst_degradation_threshold': 0.05
        }

        self.mol_cache = {}
        self.steric_cache = {}
        self.catalyst_cache = {}

    def is_valid_smiles(self, smiles):
        if not smiles or str(smiles).lower() in ['nan', 'none']:
            return False
        mol = Chem.MolFromSmiles(str(smiles))
        return mol is not None

    def extract_numbers_from_smiles(self, smiles):
        numbers = re.findall(r'\d+', smiles)
        return sum(int(num) for num in numbers) if numbers else 0

    def apply_domain_knowledge(self, prediction, conditions):
        try:
            base_yield = min(100, max(0, float(prediction)))
            
            temp = float(conditions.get('temp', 80))
            time = float(conditions.get('time', 2))
            catalyst_qty = float(conditions.get('quantity', 0.01))
            subs1_steric = float(conditions.get('subs1_steric', 0))
            subs2_steric = float(conditions.get('subs2_steric', 0))
            catalyst_smiles = conditions.get('catalyst_smiles', '')
            
            temp_diff = (temp - self.chemical_intuition['optimal_temp']) / self.chemical_intuition['temp_range']
            temp_effect = 8 * math.exp(-2.5 * temp_diff**2)
            
            if temp > self.chemical_intuition['temp_degradation_threshold']:
                excess = temp - self.chemical_intuition['temp_degradation_threshold']
                temp_effect -= 0.15 * excess**1.5
            elif temp < self.chemical_intuition['temp_too_low_threshold']:
                deficit = self.chemical_intuition['temp_too_low_threshold'] - temp
                temp_effect -= 0.1 * deficit
            
            time_diff = (time - self.chemical_intuition['optimal_time']) / self.chemical_intuition['time_range']
            time_effect = 44 * math.exp(-2 * time_diff**2)
            
            if time > 24:
                excess_time = time - 24
                time_effect -= 0.2 * excess_time
            
            catalyst_effect = 0
            if catalyst_qty > 0:
                normalized_qty = min(catalyst_qty, self.chemical_intuition['max_catalyst'])
                base_effect = self.chemical_intuition['catalyst_v_max'] * (normalized_qty / (normalized_qty + self.chemical_intuition['catalyst_k_m']))
                quality_score = self.calculate_catalyst_score(catalyst_smiles)
                quality_bonus = self.chemical_intuition['quality_bonus'] * math.log(1 + quality_score) *2
                catalyst_effect = 4*(base_effect+ quality_bonus)
                
                if catalyst_qty > self.chemical_intuition['catalyst_degradation_threshold']:
                    excess_catalyst = catalyst_qty - self.chemical_intuition['catalyst_degradation_threshold']
                    catalyst_effect -= 0.8 * excess_catalyst

                catalyst_effect -= catalyst_effect * 0.1
            
            steric_effect = 0
            max_steric = max(subs1_steric, subs2_steric)
            if max_steric > self.chemical_intuition['steric_threshold']:
                x = max_steric - self.chemical_intuition['steric_threshold']
                steric_effect = -self.chemical_intuition['steric_penalty'] / (1 + math.exp(-3*(x-0.3)))
            
            numbers_effect = 0
            subs1_smiles = conditions.get('subs1_smiles', '')
            subs2_smiles = conditions.get('subs2_smiles', '')
            
            if self.is_valid_smiles(subs1_smiles):
                numbers_count = self.extract_numbers_from_smiles(subs1_smiles)
                numbers_effect += numbers_count * 0.04
                
            if self.is_valid_smiles(subs2_smiles):
                numbers_count = self.extract_numbers_from_smiles(subs2_smiles)
                numbers_effect += numbers_count * 0.04
            
            adjusted_yield = base_yield + temp_effect + time_effect + catalyst_effect + steric_effect + numbers_effect
            final_yield = adjusted_yield + 0.2 * (2*np.random.random() - 1)
            
            final_yield = min(self.chemical_intuition['max_yield'], max(self.chemical_intuition['min_yield'], final_yield)) 
            return round(final_yield, 1)
            
        except Exception as e:
            app.logger.error(f"Error in apply_domain_knowledge: {str(e)}")
            return round(prediction, 1)

    def calculate_steric_score(self, mol):
        if not mol:
            return 0
            
        smiles = Chem.MolToSmiles(mol)
        if smiles in self.steric_cache:
            return self.steric_cache[smiles]
        
        try:
            ring_info = mol.GetRingInfo()
            atoms_in_rings = set()
            for ring in ring_info.AtomRings():
                atoms_in_rings.update(ring)
                
            ring_atom_ratio = len(atoms_in_rings) / mol.GetNumHeavyAtoms() if mol.GetNumHeavyAtoms() > 0 else 0
            bulky_groups = 0
            
            bulky_patterns = [
                '[CH3][CH3]',
                '[CH3][CH3][CH3]',
                'c1ccccc1',
                '[NH2]',
                '[OH]',
                '[F,Cl,Br,I]',
            ]
            
            for pattern in bulky_patterns:
                bulky_groups += len(mol.GetSubstructMatches(Chem.MolFromSmarts(pattern)))
            
            steric_score = (1 - ring_atom_ratio) * 0.6 + min(1.0, bulky_groups * 0.2)
            
            self.steric_cache[smiles] = min(1.0, max(0.0, steric_score))
            return self.steric_cache[smiles]
            
        except Exception as e:
            app.logger.error(f"Error calculating steric score: {str(e)}")
            return 0

    def calculate_catalyst_score(self, catalyst_smiles):
        if not catalyst_smiles or str(catalyst_smiles).lower() in ['nan', 'none']:
            return 0
            
        if catalyst_smiles in self.catalyst_cache:
            return self.catalyst_cache[catalyst_smiles]
        
        try:
            mol = Chem.MolFromSmiles(str(catalyst_smiles))
            if not mol:
                return 0
                
            score = 0
            
            electronegative_atoms = {'N', 'O', 'F', 'P', 'S', 'Cl', 'Br'}
            electron_donor_groups = ['NH2', 'OH', 'OCH3', 'NMe2', 'OMe', 'PPh3', 'PMe3']
            metal_centers = ['Pd', 'Pt', 'Ni', 'Cu']
            
            for atom in mol.GetAtoms():
                if atom.GetSymbol() in metal_centers:
                    score += 5.0
                    break
                    
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol in electronegative_atoms:
                    score += 1.0
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'C':
                            score += 0.3
                            
            for group in electron_donor_groups:
                if mol.HasSubstructMatch(Chem.MolFromSmiles(group)):
                    score += 1.5
                    
            normalized_score = min(score, 10)
            self.catalyst_cache[catalyst_smiles] = normalized_score
            return normalized_score
            
        except Exception as e:
            app.logger.error(f"Error calculating catalyst score: {str(e)}")
            return 0

    def load_and_prepare_data(self, path):
        try:
            self.original_df = pd.read_csv(path)
            self.original_df.columns = self.original_df.columns.str.lower().str.replace(' ', '_')
            df = self.original_df.copy()
            
            if 'yield' in df.columns:
                df['yield'] = pd.to_numeric(df['yield'], errors='coerce')
                df = df.dropna(subset=['yield'])
                df['yield'] = df['yield'].clip(0, 100)
                
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].mean())
                
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                df[col] = df[col].astype(str)
                df[col] = df[col].replace(['nan', 'None', 'NaT', 'N/A', 'NA', ''], 'missing')
                df[col] = df[col].fillna('missing')
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                
            for i in [1, 2]:
                if f'subs{i}_smiles' in df.columns:
                    df[f'subs{i}_mol'] = df[f'subs{i}_smiles'].apply(lambda x: Chem.MolFromSmiles(str(x)) if pd.notna(x) and str(x) not in ['nan', 'None'] else None)
                    df[f'subs{i}_ha'] = df[f'subs{i}_mol'].apply(lambda x: x.GetNumHeavyAtoms() if x else np.nan)
                    df[f'subs{i}_rb'] = df[f'subs{i}_mol'].apply(lambda x: Descriptors.NumRotatableBonds(x) if x else np.nan)
                    df[f'subs{i}_logp'] = df[f'subs{i}_mol'].apply(lambda x: Descriptors.MolLogP(x) if x else np.nan)
                    df[f'subs{i}_ring_count'] = df[f'subs{i}_mol'].apply(lambda x: x.GetRingInfo().NumRings() if x else np.nan)
                    df[f'subs{i}_steric'] = df[f'subs{i}_mol'].apply(lambda x: self.calculate_steric_score(x) if x else np.nan)
                    
            new_numeric = ['subs1_ha', 'subs2_ha', 'subs1_rb', 'subs2_rb', 'subs1_logp', 'subs2_logp', 'subs1_ring_count', 'subs2_ring_count', 'subs1_steric', 'subs2_steric']
            for col in new_numeric:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mean())
                    
            self.df = df
            return True, f"Data loaded successfully: {len(df)} rows"
        except Exception as e:
            traceback.print_exc()
            return False, f"Data loading error: {str(e)}"

    def select_model(self, model_name):
        if model_name not in self.models:
            return False, f"Unknown model: {model_name}"
        if self.df is None:
            return False, "You need to load data first"
        try:
            self.current_model_name = model_name
            numeric_features = ['temp', 'time', 'quantity', 'subs1_ha', 'subs2_ha', 'subs1_rb', 'subs2_rb', 'subs1_logp', 'subs2_logp', 'subs1_ring_count', 'subs2_ring_count', 'subs1_steric', 'subs2_steric']
            categorical_features = ['catalizor', 'base', 'solv1', 'solv2']
            available_features = []
            for feature in numeric_features + categorical_features:
                if feature in self.df.columns:
                    available_features.append(feature)
            if len(available_features) < 3:
                return False, "Not enough features available"
            available_numeric = [f for f in numeric_features if f in available_features]
            available_categorical = [f for f in categorical_features if f in available_features]
            if model_name in ['SVR', 'Neural Network']:
                self.model = self.models[model_name]
            else:
                numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
                categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value=-1)), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
                preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, available_numeric), ('cat', categorical_transformer, available_categorical)])
                self.model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', self.models[model_name])])
            X = self.df[available_features]
            y = self.df['yield']
            self.feature_names = list(X.columns)
            self.model.fit(X, y)
            y_pred = self.model.predict(X)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            return True, f"{model_name} model trained (MAE: {mae:.2f}, R²: {r2:.2f})"
        except Exception as e:
            traceback.print_exc()
            return False, f"Model training error: {str(e)}"

    def get_original_categories(self, column_name):
        if column_name in self.label_encoders:
            return list(self.label_encoders[column_name].classes_)
        elif self.original_df is not None and column_name in self.original_df.columns:
            return sorted(self.original_df[column_name].astype(str).unique().tolist())
        return []

    def predict_yield(self, reaction_conditions):
        if self.model is None:
            return None, "Model not trained yet"
        try:
            # Validate SMILES inputs
            for i in [1, 2]:
                subs_key = f'subs{i}_smiles'
                if subs_key in reaction_conditions:
                    smiles = reaction_conditions[subs_key]
                    if not self.is_valid_smiles(smiles):
                        return None, f"Invalid SMILES string for {subs_key}: {smiles}"
            
            input_df = pd.DataFrame([reaction_conditions])
            for col in input_df.select_dtypes(include=['object']).columns:
                if col in self.label_encoders:
                    input_df[col] = input_df[col].astype(str)
                    input_df[col] = input_df[col].replace(['nan', 'None', 'NaT', 'N/A', 'NA', ''], 'missing')
                    try:
                        input_df[col] = self.label_encoders[col].transform(input_df[col])
                    except ValueError:
                        input_df[col] = len(self.label_encoders[col].classes_)
                        
            for i in [1, 2]:
                if f'subs{i}_smiles' in reaction_conditions:
                    smiles = reaction_conditions[f'subs{i}_smiles']
                    if smiles and str(smiles) not in ['nan', 'None']:
                        mol = Chem.MolFromSmiles(str(smiles))
                        if mol:
                            input_df[f'subs{i}_ha'] = mol.GetNumHeavyAtoms()
                            input_df[f'subs{i}_rb'] = Descriptors.NumRotatableBonds(mol)
                            input_df[f'subs{i}_logp'] = Descriptors.MolLogP(mol)
                            input_df[f'subs{i}_ring_count'] = mol.GetRingInfo().NumRings()
                            input_df[f'subs{i}_steric'] = self.calculate_steric_score(mol)
                            
            for col in input_df.columns:
                if pd.api.types.is_numeric_dtype(input_df[col]):
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                    input_df[col] = input_df[col].fillna(input_df[col].mean())
                else:
                    input_df[col] = input_df[col].fillna(-1)
                    
            input_df = input_df[self.feature_names] if self.feature_names else input_df
            prediction = float(self.model.predict(input_df)[0])
            adjusted_prediction = self.apply_domain_knowledge(prediction, reaction_conditions)
            if isinstance(adjusted_prediction, str) and adjusted_prediction.startswith("ERROR:"):
                return None, adjusted_prediction
            final_prediction = max(10, min(98, round(adjusted_prediction, 1)))
            return final_prediction, None
        except Exception as e:
            traceback.print_exc()
            return None, f"Prediction error: {str(e)}"

    def _get_catalyst_smiles_mapping(self):
        if self.original_df is None:
            return {}
        mapping = {}
        if 'catalizor' in self.original_df.columns and 'catalyst_smiles' in self.original_df.columns:
            for _, row in self.original_df.iterrows():
                if pd.notna(row['catalizor']) and pd.notna(row['catalyst_smiles']):
                    mapping[row['catalizor']] = row['catalyst_smiles']
        return mapping

    def optimize_catalyst(self, reaction_conditions):
        if self.df is None or 'catalizor' not in self.df.columns:
            return None, "Data not loaded or no catalyst info"
        try:
            # Validate SMILES inputs first
            for i in [1, 2]:
                subs_key = f'subs{i}_smiles'
                if subs_key in reaction_conditions:
                    smiles = reaction_conditions[subs_key]
                    if not self.is_valid_smiles(smiles):
                        return None, f"Invalid SMILES string for {subs_key}: {smiles}"
                        
            catalysts = self.get_original_categories('catalizor')
            catalyst_smiles_map = self._get_catalyst_smiles_mapping()
            results = []
            for catalyst in catalysts[:20]:
                conditions = reaction_conditions.copy()
                conditions['catalizor'] = catalyst
                conditions['catalyst_smiles'] = catalyst_smiles_map.get(catalyst, '')
                yield_pred, error = self.predict_yield(conditions)
                if yield_pred is not None:
                    catalyst_score = self.calculate_catalyst_score(conditions['catalyst_smiles'])
                    adjusted_yield = yield_pred + 0.5 * catalyst_score
                    results.append((catalyst, min(100, adjusted_yield), conditions['catalyst_smiles']))
            sorted_results = sorted(results, key=lambda x: (-x[1], -self.calculate_catalyst_score(x[2])))
            return [(x[0], round(x[1],1)) for x in sorted_results[:5]], None
        except Exception as e:
            traceback.print_exc()
            return None, f"Optimization error: {str(e)}"

    def get_top_catalysts_from_data(self, n=5):
        if self.original_df is None or 'catalizor' not in self.original_df.columns:
            return None
        return (self.original_df.groupby('catalizor')['yield'].agg(['mean', 'count', 'std']).sort_values('mean', ascending=False).head(n))

    def visualize_molecules(self, smiles_list, legends=None):
        try:
            mols = [Chem.MolFromSmiles(str(smiles)) for smiles in smiles_list if smiles and str(smiles) not in ['nan', 'None']]
            valid_mols = [m for m in mols if m is not None]
            if not valid_mols:
                return None
            if legends is None or len(legends) != len(smiles_list):
                legends = [f"Substrate {i+1}" for i in range(len(smiles_list))]
            valid_legends = [legends[i] for i, m in enumerate(mols) if m is not None]
            img = Draw.MolsToGridImage(valid_mols, legends=valid_legends, molsPerRow=min(2, len(valid_mols)), subImgSize=(300, 300))
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str
        except Exception as e:
            traceback.print_exc()
            print(f"Visualization error: {str(e)}")
            return None

predictor = SuzukiReactionPredictor()

def handle_errors(f):
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            app.logger.error(f"Error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    return wrapper

def standardize_smiles(smiles):
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            Chem.Kekulize(mol)
            return Chem.MolToSmiles(mol, kekuleSmiles=True)
        return None
    except Exception as e:
        print(f"SMILES standardization error: {str(e)}")
        return None

def get_pubchem_smiles(name):
    if not name or str(name).lower() == 'nan':
        return None
    
    name = str(name).strip()
    if name in smiles_cache:
        return smiles_cache[name]
    
    try:
        name_encoded = requests.utils.quote(name)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name_encoded}/property/CanonicalSMILES/JSON"
        response = requests.get(url, timeout=10)
        
        if 'application/json' not in response.headers.get('Content-Type', ''):
            raise ValueError("PubChem API invalid response format")
            
        data = response.json()
        smiles = data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
        smiles_cache[name] = smiles
        return smiles
    except Exception as e:
        print(f"PubChem error for {name}: {str(e)}")
        return None

def execute_command(command):
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True, encoding=locale.getpreferredencoding(), errors='replace')
        stdout, stderr = process.communicate()
        
        def preserve_colors(text):
            return text.replace('\x1b', '\x1b')
        
        if process.returncode != 0:
            return False, preserve_colors(stderr if stderr else "Command failed")
        return True, preserve_colors(stdout)
    except Exception as e:
        return False, str(e)

def merge_csv_to_output(input_files, output_file='out.csv'):
    global rowcount
    try:
        a=[]
        for x in input_files:
            a.append("static/datasets/"+x)
        dfs = [pd.read_csv(f) for f in a]
        merged_df = pd.concat(dfs, ignore_index=True)
        rowcount = len(merged_df)
        merged_df.to_csv(output_file, index=False)
        asdf=os.path.join('static/model', 'miaw.csv')
        merged_df.to_csv(asdf, index=False)
        return rowcount
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        return 0

def get_rowcount_for_datasets(dataset_files):
    try:
        total_rows = 0
        for dataset_file in dataset_files:
            file_path = os.path.join("static/datasets", dataset_file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                total_rows += len(df)
        return total_rows
    except Exception as e:
        print(f"Satır sayısı hesaplanırken hata: {str(e)}")
        return 0

def get_datasets_from_folder(folder_path):
    datasets = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    datasets = [f for f in datasets if f.endswith('.csv')]
    return datasets

def get_codes_from_folder(folder_path):
    codes = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    codes = [f for f in codes if f.endswith('.py')]
    return codes

def run_python_script(script_name, datasets):
    command = ['python', f'code/{script_name}'] + datasets
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    for line in process.stdout:
        app.logger.info(line.strip())
    process.wait()

def sanitize_csv_filename(filename):  
    filename = os.path.basename(filename) 
    if not filename.lower().endswith('.csv'):
        filename += '.csv'
    filename = re.sub(r'[^\w\-_.]', '_', filename)
    return filename

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/analyse_dataset')
def analyze_dataset():
    return render_template('analyse_dataset.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file selected'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        if file and file.filename.endswith('.csv'):
            filename = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            filepath = os.path.join(OUTPUTS_DIR, filename)
            
            if not os.path.exists(OUTPUTS_DIR):
                os.makedirs(OUTPUTS_DIR)
            
            file.save(filepath)
            return jsonify({'success': True, 'message': f'File successfully uploaded: {filename}'})
        else:
            return jsonify({'success': False, 'message': 'Only CSV files can be uploaded'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Upload error: {str(e)}'})

@app.route('/progress/<task_id>')
def get_progress(task_id):
    if task_id not in progress_data:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(progress_data[task_id])

@app.route('/download/<path:filename>')
def download_file(filename):
    try:
        safe_path = os.path.join(app.root_path, filename)
        if not os.path.exists(safe_path):
            return jsonify({'error': 'File not found'}), 404
        return send_file(safe_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
@handle_errors
def process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'}), 400
    
    use_rdkit = request.form.get('use_rdkit', 'false') == 'true'
    use_pubchem = request.form.get('use_pubchem', 'false') == 'true'
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return jsonify({'error': f'CSV read error: {str(e)}'}), 400
    
    for col in df.columns:
        if df[col].dtype == object:
            if use_rdkit:
                df[col] = df[col].apply(standardize_smiles)
            if use_pubchem:
                df[col] = df[col].apply(get_pubchem_smiles)
    
    output_filename = f"processed_{filename}"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    df.to_csv(output_path, index=False)
    
    return jsonify({
        'success': True,
        'download_url': f'/download/{output_filename}',
        'preview': df.head(10).to_html(classes='table table-striped')
    })

@app.route('/csv_clean')
def csv_clean():
    return render_template('csv_processor.html')

@app.route('/terminal')
def terminal():
    return render_template('kratos.html')

@app.route('/predict')
def predict():
    global rowcount
    datasets = get_datasets_from_folder(DATASET_LOG_DIR2)
    codess = get_codes_from_folder(CODE_LOG_DIR)
    app.logger.info("Accessed the main index page.")
    return render_template('predict.html', datasets=datasets, codess=codess, rowcount=rowcount)

@app.route('/get_dataset_rowcount', methods=['POST'])
def get_dataset_rowcount():
    data = request.get_json()
    selected_datasets = data.get('selectedDatasets', [])
    if not selected_datasets:
        return jsonify({"rowcount": 0})
    total_rows = get_rowcount_for_datasets(selected_datasets)
    return jsonify({"rowcount": total_rows})

@app.route('/visual')
def visual():
    folders = [f.name for f in os.scandir(IMAGES_LOG_DIR) if f.is_dir()]
    return render_template('visual.html', images=folders)

@app.route('/run_model', methods=['POST'])
def run_model():
    data = request.get_json()
    selected_datasets = data.get('selectedDatasets', [])
    selected_code = data.get('selectedCode', '')
    global rowcount
    rowcount = merge_csv_to_output(selected_datasets,"out.csv")
    if not selected_code:
        return jsonify({"status": "error", "message": "No model script selected."}), 400
    app.logger.info(f"Starting model: {selected_code} with datasets: {selected_datasets}")
    thread = threading.Thread(target=run_python_script, args=(selected_code, selected_datasets))
    thread.start()
    return jsonify({"status": "success", "message": "Model is running!"})

@app.route('/stream_output')
def stream_output():
    selected_code = request.args.get('code')
    def generate():
        if not selected_code:
            yield "data: ERROR: No script selected\n\n"
            return
        command = ['python', f'code/{selected_code}']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
        while True:
            output = process.stdout.readline()
            if output:
                yield f"data: {output.strip()}\n\n"
            error = process.stderr.readline()
            if error:
                yield f"data: ERROR: {error.strip()}\n\n"
            if process.poll() is not None: 
                for output in process.stdout:
                    if output: yield f"data: {output.strip()}\n\n"
                for error in process.stderr:
                    if error: yield f"data: ERROR: {error.strip()}\n\n"
                yield "data: COMPLETE: Model training finished\n\n"
                break
    return Response(generate(), content_type='text/event-stream')

@app.route('/get_latest_result')
def get_latest_result():
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        result_files = glob.glob(os.path.join(RESULTS_DIR, '*.txt'))
        if not result_files:
            return jsonify({"status": "error", "message": "No results found"}), 404
        latest_file = max(result_files, key=os.path.getctime)
        with open(latest_file, 'r') as f:
            content = f.read()
        return jsonify({
            "status": "success",
            "content": content,
            "filename": os.path.basename(latest_file)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/download_model')
def download_model():
    model_path = os.path.join(TRAINED_MODELS_DIR, 'models.pkl')
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    return jsonify({"status": "error", "message": "Model file not found"}), 404

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/edit_dataset')
def edit_dataset():
    return render_template('edit.html')

@app.route('/preview_dataset')
def preview_dataset():
    return render_template('preview.html')

@app.route('/excel_to_csv')
def excel_to_csv():
    return render_template('xlsx.html')

@app.route('/manual')
def manuel():
    return render_template('manual.html')

@app.route('/upload-csv')
def upload_csv():
    return render_template('csv.html')

@app.route('/save_data', methods=['POST'])
def save_data():
    try:
        data = request.get_json()
        filename = data.get('filename', 'data')
        content = data.get('data', '')
        os.makedirs('static/datasets', exist_ok=True)
        filepath = os.path.join('static/datasets', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return jsonify({'success': True, 'message': 'File saved successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/compare-models')
def compare_models():
    return render_template('messi.html')

@app.route('/get_xml_files')
def get_xml_files():
    xml_files = []
    try:
        xml_dir = os.path.join(app.root_path, 'xml_data')
        os.makedirs(xml_dir, exist_ok=True)
        for file in glob.glob(os.path.join(xml_dir, '*.xml')):
            xml_files.append(os.path.basename(file))
    except Exception as e:
        app.logger.error(f"XML dosyaları listelenirken hata: {str(e)}")
    return jsonify(xml_files)

@app.route('/get_xml_data')
def get_xml_data():
    filename = request.args.get('file', '')
    if not filename:
        return jsonify({"error": "Dosya adı belirtilmedi"}), 400
    try:
        if not filename.lower().endswith('.xml'):
            return jsonify({"error": "Geçersiz dosya türü"}), 400
        filepath = os.path.join(app.root_path, 'xml_data', filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "Dosya bulunamadı"}), 404
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/xml'}
    except Exception as e:
        app.logger.error(f"XML dosyası okunurken hata: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/save-csv', methods=['POST'])
def save_csv():
    try:
        data = request.get_json()
        csv_content = data.get('csv', '')
        filenamea = data.get('filename', '')
        if not csv_content:
            return jsonify({'success': False, 'message': 'Boş CSV içeriği'})
        save_path = os.path.join(DATASET_LOG_DIR2, filenamea)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        return jsonify({'success': True, 'message': f'CSV başarıyla kaydedildi: {filenamea}'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata oluştu: {str(e)}'})

@app.route('/api/load-csv', methods=['POST'])
def api_load_csv():
    data = request.get_json()
    url = data.get('url', '')
    if not url:
        return jsonify({'success': False, 'message': 'URL is required'}), 400
    try:
        response = requests.get(url)
        response.raise_for_status()
        return jsonify({'success': True, 'content': response.text, 'filename': os.path.basename(urlparse(url).path)})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/save-csv', methods=['POST'])
def api_save_csv():
    try:
        data = request.get_json()
        content = data.get('content', '')
        filename = data.get('filename', '')
        if not content:
            return jsonify({'success': False, 'message': 'No content provided'}), 400
        filename = sanitize_csv_filename(filename)
        print(filename)
        os.makedirs(DATASET_LOG_DIR2, exist_ok=True)
        filepath = os.path.join(DATASET_LOG_DIR2, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return jsonify({
            'success': True,
            'message': 'CSV saved successfully',
            'filename': filename,
            'path': f'static/datasets/{filename}'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/execute', methods=['POST'])
def api_execute():
    data = request.json
    command = data.get('command', '').strip()
    
    if not command:
        return jsonify({'success': False, 'output': 'No command provided'})
    
    success, output = execute_command(command)
    return jsonify({
        'success': success,
        'output': output,
        'cwd': os.getcwd()
    })

@app.route('/api/getcwd', methods=['GET'])
def api_getcwd():
    return jsonify({'cwd': os.getcwd()})

@app.route('/predict_ml')
def predict_ml_page():
    csv_files = [f for f in os.listdir('static/datasets') if f.endswith('.csv')] if os.path.exists('static/datasets') else []
    models = list(predictor.models.keys())
    current_model = predictor.current_model_name
    has_data = predictor.original_df is not None
    
    data_info = {}
    if has_data:
        data_info = {
            'catalysts': predictor.get_original_categories('catalizor'),
            'bases': predictor.get_original_categories('base'),
            'solvents1': predictor.get_original_categories('solv1'),
            'solvents2': predictor.get_original_categories('solv2')
        }
    
    return render_template('predict_ml.html', csv_files=csv_files, models=models, current_model=current_model, has_data=has_data, data_info=data_info)

@app.route('/api/get_csv_files')
def api_get_csv_files():
    try:
        output_dir = os.path.join(app.root_path, OUTPUTS_DIR)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            return jsonify({'success': True, 'files': []})
        
        csv_files = []
        for filename in os.listdir(output_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(output_dir, filename)
                try:
                    size = os.path.getsize(filepath)
                    
                    if size < 1024:
                        size_str = f"{size} B"
                    elif size < 1024 * 1024:
                        size_str = f"{size // 1024} KB"
                    else:
                        size_str = f"{size // (1024 * 1024)} MB"
                    
                    csv_files.append({
                        'name': filename,
                        'size': size_str
                    })
                except Exception as e:
                    app.logger.error(f"Error processing file {filename}: {str(e)}")
                    continue
        
        csv_files.sort(key=lambda x: x['name'])
        return jsonify({'success': True, 'files': csv_files})
    except Exception as e:
        app.logger.error(f"Error in get_csv_files: {str(e)}")
        return jsonify({'success': False, 'message': f'Could not get file list: {str(e)}'})

@app.route('/load_data', methods=['POST'])
def load_data():
    try:
        filename = request.json.get('filename')
        if not filename:
            return jsonify({'success': False, 'message': 'Filename not specified'})
        
        filepath = os.path.join('static/datasets', filename)
    
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'File not found'})
        
        success, message = predictor.load_and_prepare_data(filepath)
        if success:
            model_success, model_message = predictor.select_model(predictor.current_model_name)
            
            data_info = {
                'rows': len(predictor.original_df),
                'columns': list(predictor.original_df.columns),
                'catalysts': predictor.get_original_categories('catalizor'),
                'bases': predictor.get_original_categories('base'),
                'solvents1': predictor.get_original_categories('solv1'),
                'solvents2': predictor.get_original_categories('solv2')
            }
            
            return jsonify({
                'success': success, 
                'message': f"{message} - {model_message}",
                'data_info': data_info
            })
        else:
            return jsonify({'success': False, 'message': message})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/api/model_performance')
def api_model_performance():
    try:
        if predictor.df is None:
            return jsonify({'success': False, 'message': 'Data not loaded'})
        
        stats = {
            'data_size': len(predictor.df),
            'current_model': predictor.current_model_name,
            'yield_mean': float(predictor.df['yield'].mean()),
            'yield_std': float(predictor.df['yield'].std()),
            'yield_min': float(predictor.df['yield'].min()),
            'yield_max': float(predictor.df['yield'].max())
        }
        
        if predictor.model is not None and hasattr(predictor.model, 'predict'):
            X = predictor.df[[col for col in predictor.df.columns if col != 'yield']]
            y = predictor.df['yield']
            y_pred = predictor.model.predict(X)
            stats['mae'] = float(mean_absolute_error(y, y_pred))
            stats['r2'] = float(r2_score(y, y_pred))
        
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Statistics error: {str(e)}'})

@app.route('/api/change_model', methods=['POST'])
def api_change_model():
    try:
        if not request.is_json:
            return jsonify({'success': False, 'message': 'Request must be JSON'}), 400

        data = request.get_json()
        
        if not data or 'model_name' not in data:
            return jsonify({'success': False, 'message': 'Model name is required'}), 400
            
        model_name = data['model_name']
        
        success, message = predictor.select_model(model_name)
        
        if not success:
            return jsonify({'success': False, 'message': message}), 400

        response_data = {
            'success': True,
            'message': message,
            'current_model': predictor.current_model_name
        }

        if hasattr(predictor, 'df') and predictor.df is not None:
            X = predictor.df[[col for col in predictor.df.columns if col != 'yield']]
            y = predictor.df['yield']
            y_pred = predictor.model.predict(X)
            response_data['stats'] = {
                'mae': float(mean_absolute_error(y, y_pred)),
                'r2': float(r2_score(y, y_pred))
            }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500

@app.route('/api/make_prediction', methods=['POST'])
def api_make_prediction():
    try:
        conditions = request.get_json()
        prediction, error = predictor.predict_yield(conditions)
        
        if error:
            return jsonify({'success': False, 'message': error}), 400
        
        smiles_list = [conditions.get('subs1_smiles', ''), conditions.get('subs2_smiles', '')]
        img_str = predictor.visualize_molecules(smiles_list, ['Boronic Acid', 'Aryl Halide'])
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'model': predictor.current_model_name,
            'molecule_image': img_str
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Prediction error: {str(e)}'}), 500

@app.route('/api/optimize_catalyst', methods=['POST'])
def api_optimize_catalyst():
    try:
        conditions = request.get_json()
        results, error = predictor.optimize_catalyst(conditions)
        
        if error:
            return jsonify({'success': False, 'message': error}), 400
        
        smiles_list = [conditions.get('subs1_smiles', ''), conditions.get('subs2_smiles', '')]
        img_str = predictor.visualize_molecules(smiles_list, ['Boronic Acid', 'Aryl Halide'])
        
        return jsonify({
            'success': True,
            'results': results,
            'model': predictor.current_model_name,
            'molecule_image': img_str
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Optimization error: {str(e)}'}), 500

@app.route('/api/get_top_catalysts')
def api_get_top_catalysts():
    try:
        top_catalysts = predictor.get_top_catalysts_from_data()
        if top_catalysts is not None:
            return jsonify({
                'success': True,
                'data': top_catalysts.to_dict('index')
            })
        else:
            return jsonify({'success': False, 'message': 'Data not loaded or no catalyst info'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/api/model_comparison', methods=['POST'])
def api_model_comparison():
    try:
        filename = request.json.get('filename')
        if not filename:
            return jsonify({'success': False, 'message': 'Filename not specified'})
        
        filepath = os.path.join('outputs', filename)
        
        temp_predictor = SuzukiReactionPredictor()
        success, message = temp_predictor.load_and_prepare_data(filepath)
        
        if not success:
            return jsonify({'success': False, 'message': message})
        
        results = {}
        
        for model_name in temp_predictor.models.keys():
            try:
                success, _ = temp_predictor.select_model(model_name)
                if success:
                    X = temp_predictor.df[[col for col in temp_predictor.df.columns if col != 'yield']]
                    y = temp_predictor.df['yield']
                    
                    cv_scores = cross_val_score(temp_predictor.model, X, y, cv=5, scoring='r2')
                    
                    results[model_name] = {
                        'mean_score': float(cv_scores.mean()),
                        'std_score': float(cv_scores.std()),
                        'scores': cv_scores.tolist()
                    }
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Model comparison error: {str(e)}'})

@app.route('/api/upload_csv', methods=['POST'])
def api_upload_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file selected'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        if file and file.filename.endswith('.csv'):
            filename = f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            filepath = os.path.join('outputs', filename)
            
            if not os.path.exists('outputs'):
                os.makedirs('outputs')
            
            file.save(filepath)
            return jsonify({'success': True, 'message': f'File successfully uploaded: {filename}'})
        else:
            return jsonify({'success': False, 'message': 'Only CSV files can be uploaded'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Upload error: {str(e)}'})

if __name__ == '__main__':
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    app.run(debug=True, host='0.0.0.0', port=5000)
