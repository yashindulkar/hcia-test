from django.shortcuts import render
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import pandas as pd
import json
import sys
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import seaborn as sns
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import tempfile

# Default view for initial page load
from sklearn.linear_model import LogisticRegression

def index(request):
    # Get lambda value from request or use default
    lambda_value = request.GET.get('lambda', 0.01)
    try:
        lambda_value = float(lambda_value)
    except ValueError:
        lambda_value = 0.01

    # Load Palmer Penguins dataset
    penguins_data = load_penguins_data()
    if penguins_data is None:
        return render(request, 'project3/index.html', {
            'error': 'Failed to load the Palmer Penguins dataset. Make sure palmerpenguins is installed.'
        })

    # Prepare data
    X, y, feature_names, target_names, X_raw = prepare_data(penguins_data)
    X_train, X_test, y_train, y_test, X_train_raw, X_test_raw = train_test_split(
        X, y, X_raw, test_size=0.3, random_state=42
    )

    # ------------------- Decision Tree -------------------
    max_depth = max(2, min(10, int(10 - lambda_value * 9)))
    tree_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree_model.fit(X_train, y_train)

    tree_y_pred = tree_model.predict(X_test)
    tree_accuracy = accuracy_score(y_test, tree_y_pred)
    tree_leaves = tree_model.get_n_leaves()

    tree_plot = visualize_decision_tree(tree_model, feature_names, target_names)
    importance_plot = plot_feature_importance(tree_model, feature_names)

    # ------------------- Logistic Regression -------------------
    C_value = max(0.001, 1.0 / lambda_value)  # Regularization strength for sparsity
    logistic_model = LogisticRegression(
        penalty='l1',
        solver='saga',
        multi_class='multinomial',
        C=C_value,
        max_iter=1000,
        random_state=42
    )
    logistic_model.fit(X_train, y_train)

    logistic_y_pred = logistic_model.predict(X_test)
    logistic_accuracy = accuracy_score(y_test, logistic_y_pred)
    logistic_used_features = (logistic_model.coef_ != 0).sum(axis=1).max()

    # Generate coefficient plot for Logistic Regression
    logistic_plot = plot_logistic_coefficients(logistic_model, feature_names)
    
    # ------------------- Correlation Heatmap (shared) -------------------
    corr_plot = create_correlation_heatmap(penguins_data)

    # ------------------- Context -------------------
    context = {
        # Shared
        'lambda_value': float(lambda_value),
        'features': feature_names,
        'target': 'Species',
        'corr_plot': corr_plot,

        # Decision Tree results
        'tree_accuracy': round(tree_accuracy * 100, 2),
        'tree_n_leaves': int(tree_leaves),
        'tree_plot': tree_plot,
        'importance_plot': importance_plot,
        'max_depth': max_depth,

        # Logistic Regression results
        'logistic_accuracy': round(logistic_accuracy * 100, 2),
        'logistic_n_features': int(logistic_used_features),
        'logistic_plot': logistic_plot,
        'C_value': round(C_value, 4)
    }

    return render(request, 'project3/index.html', context)

@csrf_exempt
def update_model(request):
    """AJAX endpoint to update both Decision Tree and Logistic Regression based on lambda value"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            lambda_value = float(data.get('lambda', 0.01))

            print(f"Received lambda value: {lambda_value}")

            # Load Palmer Penguins dataset
            penguins_data = load_penguins_data()
            if penguins_data is None:
                return JsonResponse({'error': 'Failed to load dataset'}, status=400)

            # Prepare data
            X, y, feature_names, target_names, X_raw = prepare_data(penguins_data)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # ---------------- Decision Tree ----------------
            max_depth = max(2, min(10, int(10 - lambda_value * 9)))
            tree_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            tree_model.fit(X_train, y_train)

            tree_y_pred = tree_model.predict(X_test)
            tree_accuracy = accuracy_score(y_test, tree_y_pred)
            tree_leaves = tree_model.get_n_leaves()

            tree_plot = visualize_decision_tree(tree_model, feature_names, target_names)
            importance_plot = plot_feature_importance(tree_model, feature_names)

            # ---------------- Logistic Regression ----------------
            from sklearn.linear_model import LogisticRegression
            import numpy as np

            C_value = max(0.001, 1.0 / lambda_value)
            logistic_model = LogisticRegression(
                penalty='l1',
                solver='saga',
                multi_class='multinomial',
                C=C_value,
                max_iter=1000,
                random_state=42
            )
            logistic_model.fit(X_train, y_train)

            logistic_y_pred = logistic_model.predict(X_test)
            logistic_accuracy = accuracy_score(y_test, logistic_y_pred)
            logistic_used_features = (logistic_model.coef_ != 0).sum(axis=1).max()

            logistic_plot = plot_logistic_coefficients(logistic_model, feature_names)

            # ---------------- Return Both Results ----------------
            return JsonResponse({
                'lambda': lambda_value,

                # Decision Tree outputs
                'tree_accuracy': round(tree_accuracy * 100, 2),
                'tree_n_leaves': int(tree_leaves),
                'tree_plot': tree_plot,
                'importance_plot': importance_plot,
                'max_depth': max_depth,

                # Logistic Regression outputs
                'logistic_accuracy': round(logistic_accuracy * 100, 2),
                'logistic_n_features': int(logistic_used_features),
                'logistic_plot': logistic_plot,
                'C_value': round(C_value, 4)
            })

        except Exception as e:
            print(f"Error in update_model: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)


def load_penguins_data():
    """Load the Palmer Penguins dataset."""
    try:
        from palmerpenguins import load_penguins
        penguins = load_penguins()
        return penguins
    except ImportError:
        # If palmerpenguins is not installed, fall back to a sample dataset
        try:
            import pandas as pd
            import sklearn.datasets
            
            # Try to get Iris as a fallback (for testing)
            iris = sklearn.datasets.load_iris(as_frame=True)
            df = iris['data']
            df['species'] = pd.Series(iris['target']).map({
                0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'
            })
            return df
        except Exception as e:
            print(f"Error loading fallback dataset: {e}")
            return None

def prepare_data(df):
    """Prepare the data for modeling."""
    # Get feature names
    if 'species' in df.columns:
        target_col = 'species'
    else:
        target_col = 'species'
    
    # Select only numeric columns for features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create X and y
    X_raw = df[numeric_cols].copy()  # Keep raw version for GOSDT
    
    # Handle missing values if any
    X_raw = X_raw.fillna(X_raw.mean())
    
    # Extract target
    if target_col in df.columns:
        y = df[target_col].copy()
    else:
        # Fallback if column name is different
        y = df.iloc[:, -1].copy()
    
    # Get unique target names
    target_names = sorted(y.unique())
    
    # Encode target if needed
    if not pd.api.types.is_numeric_dtype(y):
        y_map = {name: i for i, name in enumerate(target_names)}
        y = y.map(y_map)
    
    # Scale features for sklearn models
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    return X, y, numeric_cols, target_names, X_raw.values

def visualize_decision_tree(model, feature_names, class_names):
    """Create a visualization of the decision tree."""
    # Adjusted figure size to better fit laptop screens
    plt.figure(figsize=(12, 8))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=8  # Smaller font size for better fit
    )
    plt.title("Decision Tree for Palmer Penguins Dataset")
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    plt.close()  # Close the figure to free memory
    
    return base64.b64encode(image_png).decode('utf-8')

def create_correlation_heatmap(df):
    """Create a correlation heatmap of the features."""
    # Adjusted figure size to better fit laptop screens
    plt.figure(figsize=(8, 6))
    
    # Get only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation
    corr = numeric_df.corr()
    
    # Create heatmap with adjusted font size
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, annot_kws={"size": 8})
    plt.title("Feature Correlation Matrix")
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    plt.close()  # Close the figure to free memory
    
    return base64.b64encode(image_png).decode('utf-8')

def plot_feature_importance(model, feature_names):
    """Create a bar plot of feature importances."""
    # Adjusted figure size to better fit laptop screens
    plt.figure(figsize=(8, 5))
    
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right', fontsize=9)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance in Decision Tree')
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    plt.close()  # Close the figure to free memory
    
    return base64.b64encode(image_png).decode('utf-8') 


def plot_logistic_coefficients(model, feature_names):
    
    import io
    import base64

    # Calculate mean absolute coefficients across classes
    mean_coeffs = np.mean(np.abs(model.coef_), axis=0)  # shape: (n_features,)

    # Sort features by importance
    indices = np.argsort(mean_coeffs)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_coeffs = mean_coeffs[indices]

    # Create plot
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(sorted_coeffs)), sorted_coeffs)
    plt.xticks(range(len(sorted_coeffs)), sorted_features, rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Average Absolute Coefficient')
    plt.title('Feature Importance (Logistic Regression)')
    plt.tight_layout()

    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()

    # Return base64 string
    return base64.b64encode(image_png).decode('utf-8')
