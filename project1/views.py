from django.shortcuts import render
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

from django.http import JsonResponse
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import json
from .forms import CSVUploadForm, ModelTrainForm
import os
from io import StringIO
from django.conf import settings
import tempfile
import uuid

# Global variable to store the CSV data
CSV_STORAGE = {}

#Very basic view as an example
def index(request):
    return render(request, 'project1/index.html')


def upload_csv(request):
    error = None
    csv_data = None
    headers = None
    data = None
    plots = []
    row_count = 0
    filename = None
    features_json = None
    data_json = None
    target_classes_json = None
    unique_classes_json = None
    features = None
    train_form = ModelTrainForm()
    model_report = None
    
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Read the CSV file
                csv_file = request.FILES['file']
                
                # Get the filename without extension
                filename = csv_file.name
                if filename.endswith('.csv'):
                    filename = filename[:-4]  # Remove .csv extension
                
                # Check if it's a CSV file
                if not csv_file.name.endswith('.csv'):
                    error = "Please upload a CSV file."
                else:
                    # Process the CSV file
                    df = pd.read_csv(csv_file)

                    # Store CSV data in session as a string
                    request.session['uploaded_csv'] = df.to_csv(index=False)
                    
                    # Get row count for display
                    row_count = len(df)
                    
                    # Store headers and data for display
                    headers = df.columns.tolist()
                    data = df.values.tolist()  # Show all rows
                    csv_data = True

                    model_report = None
                    train_form = ModelTrainForm(request.POST or None)

                    csv_string = request.session.get('uploaded_csv', None)
                    if csv_string:
                        from io import StringIO
                        df = pd.read_csv(StringIO(csv_string))
                    else:
                        error = "No uploaded data found. Please upload a CSV file first."

                    '''
                    if 'model_type' in request.POST and train_form.is_valid():
                        model_type = train_form.cleaned_data['model_type']
                        test_size = train_form.cleaned_data['test_size']
                        
                        X = df.iloc[:, :-1]
                        y = df.iloc[:, -1]

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

                        if model_type == 'logistic':
                            model = LogisticRegression(max_iter=1000)
                        elif model_type == 'tree':
                            model = DecisionTreeClassifier()
                        elif model_type == 'svm':
                            model = SVC()
                        else:
                            model = None

                        if model:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            raw_report = classification_report(y_test, y_pred, output_dict=True)

                            # Clean up keys
                            model_report = {}
                            for label, metrics in raw_report.items():
                                if isinstance(metrics, dict):
                                    model_report[label] = {
                                        'precision': metrics.get('precision', 0),
                                        'recall': metrics.get('recall', 0),
                                        'f1_score': metrics.get('f1-score', 0),
                                    }

                            print(model_report)
                    '''

                                        
                    # Create visualizations
                    plots = create_visualizations(df)
                    
                    # Prepare data for ChartJS
                    features = headers[:-1]  # All columns except the last one
                    features_json = json.dumps(features)
                    
                    # Convert DataFrame to JSON for ChartJS
                    data_values = df.iloc[:, :-1].values.tolist()  # All rows, all columns except the last
                    data_json = json.dumps(data_values)
                    
                    # Get target classes for coloring
                    target_classes = df.iloc[:, -1].values.tolist()  # All rows, last column
                    target_classes_json = json.dumps(target_classes)
                    
                    # Get unique classes
                    unique_classes = sorted(df.iloc[:, -1].unique().tolist())
                    unique_classes_json = json.dumps(unique_classes)
                    
            except Exception as e:
                error = f"Error processing file: {str(e)}"
    else:
        form = CSVUploadForm()
    
    return render(request, 'project1/upload_csv.html', {
        'form': form,
        'error': error,
        'csv_data': csv_data,
        'headers': headers,
        'data': data,
        'plots': plots,
        'row_count': row_count,
        'filename': filename,
        'features': features,
        'features_json': features_json,
        'data_json': data_json,
        'target_classes': target_classes_json,
        'unique_classes': unique_classes_json,
        'train_form': train_form,
        'model_report': model_report,
    })


'''
def train_model(request):
    report = None
    error = None

    if request.method == 'POST':
        form = ModelTrainForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Load and parse CSV file
                csv_file = request.FILES['file']
                df = pd.read_csv(csv_file)

                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]

                test_size = form.cleaned_data['test_size']
                model_type = form.cleaned_data['model_type']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

                # Choose model
                if model_type == 'logistic':
                    model = LogisticRegression(max_iter=1000)
                elif model_type == 'tree':
                    model = DecisionTreeClassifier()
                elif model_type == 'svm':
                    model = SVC()
                else:
                    raise ValueError("Unsupported model type")

                # Train and evaluate
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True)

                # Rename keys with hyphens to underscores
                for label, metrics in report.items():
                    if isinstance(metrics, dict):
                        if 'f1-score' in metrics:
                            metrics['f1_score'] = metrics.pop('f1-score')
                        if 'precision' in metrics:
                            metrics['precision'] = metrics['precision']
                        if 'recall' in metrics:
                            metrics['recall'] = metrics['recall']


                

            except Exception as e:
                error = str(e)
    else:
        form = ModelTrainForm()

    return render(request, 'project1/train_model.html', {
        'form': form,
        'report': report,
        'error': error,
    })
'''

def create_visualizations(df):
    plots = []
    
    try:
        # Identify features and target
        features = df.iloc[:, :-1]  # All columns except the last one
        target = df.iloc[:, -1]     # Last column is the target
        
        # Get unique target values for coloring
        unique_targets = target.unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_targets)))
        
        # Create a scatter plot matrix for the first few features
        num_features = min(4, len(features.columns))
        
        # 1. Create a pair plot for the first few features
        plt.figure(figsize=(10, 8))
        for i in range(num_features):
            for j in range(num_features):
                plt.subplot(num_features, num_features, i*num_features + j + 1)
                
                if i == j:  # Diagonal: histogram
                    for t_idx, t_val in enumerate(unique_targets):
                        subset = features.iloc[:, i][target == t_val]
                        plt.hist(subset, alpha=0.5, color=colors[t_idx])
                    plt.xlabel(features.columns[i])
                else:  # Off-diagonal: scatter plot
                    for t_idx, t_val in enumerate(unique_targets):
                        mask = target == t_val
                        plt.scatter(
                            features.iloc[mask, j],
                            features.iloc[mask, i],
                            color=colors[t_idx],
                            alpha=0.5,
                            s=20
                        )
                    plt.xlabel(features.columns[j])
                    plt.ylabel(features.columns[i])
                
                plt.xticks([])
                plt.yticks([])
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        plots.append(plot_data)
        
        # 2. Create a box plot for each feature
        plt.figure(figsize=(12, 6))
        features.boxplot()
        plt.title('Feature Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        plots.append(plot_data)
        
        # 3. Target distribution
        plt.figure(figsize=(8, 6))
        target.value_counts().plot(kind='bar')
        plt.title('Target Distribution')
        plt.ylabel('Count')
        plt.xlabel('Class')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        plots.append(plot_data)
        
    except Exception as e:
        # In case of error, return empty plots list
        print(f"Error creating visualizations: {str(e)}")
    
    return plots




def train_model_ajax(request):
    if request.method == 'POST' and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        try:
            train_form = ModelTrainForm(request.POST)
            if not train_form.is_valid():
                return JsonResponse({'error': 'Invalid form data'}, status=400)

            model_type = train_form.cleaned_data['model_type']
            test_size = train_form.cleaned_data['test_size']

            # Load CSV from session
            csv_string = request.session.get('uploaded_csv')
            if not csv_string:
                return JsonResponse({'error': 'No CSV uploaded in session'}, status=400)

            df = pd.read_csv(io.StringIO(csv_string))

            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            if y.dtype == 'object' or y.dtype.name == 'category':
                le = LabelEncoder()
                y = le.fit_transform(y)

            # Scale features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            
            # Get model-specific parameters from form data
            if model_type == 'rf':
                # Get Random Forest parameters
                n_estimators = int(request.POST.get('rf_n_estimators', 100))
                max_depth = int(request.POST.get('rf_max_depth', 5))
                max_features = request.POST.get('rf_max_features', 'sqrt')
                
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42
                )
            elif model_type == 'svm':
                # Get SVM parameters
                kernel = request.POST.get('svm_kernel', 'rbf')
                C = float(request.POST.get('svm_C', 1.0))
                gamma = float(request.POST.get('svm_gamma', 0.1))
                
                model = SVC(
                    kernel=kernel,
                    C=C,
                    gamma=gamma,
                    probability=True
                )
            elif model_type == 'xgb':
                # Get XGBoost parameters
                n_estimators = int(request.POST.get('xgb_n_estimators', 100))
                learning_rate = float(request.POST.get('xgb_learning_rate', 0.01))
                max_depth = int(request.POST.get('xgb_max_depth', 3))
                
                model = XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=0.8,
                    colsample_bytree=1.0,
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                )
            else:
                return JsonResponse({'error': 'Unsupported model'}, status=400)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Convert numeric predictions back to original labels (if LabelEncoder was used)
            if 'le' in locals():  # Only if label encoding was applied
                y_pred_labels = le.inverse_transform(y_pred)
                y_test_labels = le.inverse_transform(y_test)
            else:
                y_pred_labels = y_pred
                y_test_labels = y_test

            # Classification report based on original (string) labels
            raw_report = classification_report(y_test_labels, y_pred_labels, output_dict=True)

            # Cleaned up report for frontend
            cleaned = {
                label: {
                    'precision': round(metrics.get('precision', 0), 2),
                    'recall': round(metrics.get('recall', 0), 2),
                    'f1_score': round(metrics.get('f1-score', 0), 2)
                }
                for label, metrics in raw_report.items() if isinstance(metrics, dict)
            }

            # (Optional) add predictions to response
            response = {
                'report': cleaned,
                'predictions': list(y_pred_labels),
                'truth': list(y_test_labels)
            }
            return JsonResponse({'report': cleaned})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)

# Create your views here.
