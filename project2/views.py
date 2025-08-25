from django.shortcuts import render, redirect
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from .forms import TextClassificationForm
import io
import pickle
import os
from django.conf import settings
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join(settings.BASE_DIR, 'static', 'saved_pipeline.pkl')

def index(request):
    form = TextClassificationForm()
    test_accuracy = None
    best_params = None
    best_cv_score = None
    error_message = None
    

    if request.method == "POST":
        action = request.POST.get('action')
        form = TextClassificationForm(request.POST, request.FILES)

        if action == 'load':
            try:
                with open(MODEL_PATH, 'rb') as f:
                    loaded_pipeline = pickle.load(f)

                # Load default test data (for demo)
                df = pd.read_csv('static/IMDB Dataset.csv').sample(2000, random_state=42)
                df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
                X = df['review']
                y = df['label']

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                y_pred = loaded_pipeline.predict(X_test)
                test_accuracy = f"{accuracy_score(y_test, y_pred):.2%}"
            except Exception as e:
                error_message = f"Failed to load pre-trained model: {str(e)}"

        elif form.is_valid() and action =='train':
            file = request.FILES['file']
            try: 
                df = pd.read_csv(file)
                df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

                X = df['review']
                y = df['label']

                vec_type = form.cleaned_data['vectorizer_choice']
                model_type = form.cleaned_data['model_choice']

                vectorizer = TfidfVectorizer(stop_words='english') if vec_type == 'tfidf' else CountVectorizer(stop_words='english')
        
                if model_type == 'logreg':
                    classifier = LogisticRegression(solver = 'liblinear')
                    param_grid = {
                        'vectorizer__max_features': [10000],
                        'vectorizer__ngram_range': [(1, 1), (1, 2)],
                        'clf__C': [0.01, 0.1, 1]
                    }
                elif model_type == 'svm':
                    classifier = SVC()
                    param_grid = {
                        'vectorizer__max_features': [10000],
                        'vectorizer__ngram_range': [(1, 1), (1, 2)],
                        'clf__C': [0.1, 1, 10],
                        'clf__kernel': ['linear', 'rbf']
                    }
                elif model_type == 'rf':
                    classifier = RandomForestClassifier()
                    param_grid = None
                else:
                    classifier = LogisticRegression()

                pipeline = Pipeline([
                    ('vectorizer', vectorizer),
                    ('clf', classifier)
                ])
        
        
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=True
                )

                if param_grid:
                    grid = GridSearchCV(
                        pipeline, 
                        param_grid = param_grid, 
                        cv = 2, 
                        scoring='accuracy', 
                        verbose = 1, 
                        n_jobs = 1 
                    )
                    grid.fit(X_train, y_train)
                    best_model = grid.best_estimator_
                    y_pred = best_model.predict(X_test)
                    test_accuracy = f"{accuracy_score(y_test, y_pred):.2%}"
                    best_cv_score = f"{grid.best_score_:.2%}"
                    best_params = grid.best_params_
                    model_to_save = best_model
                else:
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    test_accuracy = f"{accuracy_score(y_test, y_pred):.2%}"
                    model_to_save = pipeline
                
                with open(MODEL_PATH, 'wb') as f:
                        pickle.dump(model_to_save, f)

            except Exception as e:
                error_message = f"Error processing the given file: {str(e)}"
    
    return render(request, 'project2/train.html', {
        'form': form,
        'result': test_accuracy,
        'best_cv_score': best_cv_score,
        'best_params': best_params,
        'error': error_message,
    })

def uncertainty_sampling(probs):
    return np.argmin(np.max(probs, axis=1))

def entropy_sampling(probs):
    log_probs = np.log(probs + 1e-10)
    entropy = -np.sum(probs * log_probs, axis=1)
    return np.argmax(entropy)

def random_sampling(pool_size):
    return np.random.randint(0, pool_size)

# ✅ View 1: auto-resume or upload
def active_learning_view(request):
    session_file = os.path.join(settings.BASE_DIR, 'static', 'active_session.pkl')

    # Auto-resume if saved session exists
    if os.path.exists(session_file):
        try:
            with open(session_file, 'rb') as f:
                session_data = pickle.load(f)

            for key, value in session_data.items():
                request.session[key] = value

            return redirect('project2:active_learning_loop')

        except Exception as e:
            error_message = f"Failed to resume session: {str(e)}"
            return render(request, 'project2/active_upload.html', { 'error': error_message })

    # No saved session — show upload form
    if request.method == "POST" and request.FILES.get('file'):
        df = pd.read_csv(request.FILES['file'])
        df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        reviews = df['review'].tolist()
        labels = df['label'].tolist()

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            reviews, labels, test_size=0.2, random_state=42
        )

        request.session['X_all'] = X_train_val
        request.session['y_all'] = y_train_val
        request.session['X_test'] = X_test
        request.session['y_test'] = y_test
        request.session['labeled'] = list(range(10))
        request.session['unlabeled'] = list(range(10, len(X_train_val)))
        request.session['strategy'] = request.GET.get('strategy', 'uncertainty')
        request.session['accuracy_history'] = []

        return redirect('project2:active_learning_loop')

    return render(request, 'project2/active_upload.html')

def active_learning_loop(request):
    X_all = request.session.get('X_all')
    y_all = request.session.get('y_all')
    labeled = request.session.get('labeled')
    unlabeled = request.session.get('unlabeled')
    X_test = request.session.get('X_test')
    y_test = request.session.get('y_test')
    strategy = request.session.get('strategy')
    accuracy_history = request.session.get('accuracy_history', [])

    if not X_all or not labeled or not unlabeled:
        return redirect('project2:active_learning')

    # Prepare datasets
    X_labeled = [X_all[i] for i in labeled]
    y_labeled = [y_all[i] for i in labeled]
    X_unlabeled = [X_all[i] for i in unlabeled]

    # Train model
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(solver='liblinear'))
    ])
    model.fit(X_labeled, y_labeled)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_history.append(round(accuracy * 100, 2))
    request.session['accuracy_history'] = accuracy_history

    # Select samples for labeling
    BATCH_SIZE = 20  # number of samples to label per iteration
    probs = model.predict_proba(X_unlabeled)

    if strategy == 'uncertainty':
        scores = np.max(probs, axis=1)
        selected_indices = np.argsort(scores)[:BATCH_SIZE]
    elif strategy == 'entropy':
        log_probs = np.log(probs + 1e-10)
        entropy = -np.sum(probs * log_probs, axis=1)
        selected_indices = np.argsort(entropy)[-BATCH_SIZE:]
    else:
        selected_indices = np.random.choice(len(unlabeled), BATCH_SIZE, replace=False)

    # Update labeled/unlabeled pools
    for i in sorted(selected_indices, reverse=True):  # reverse to safely delete
        labeled.append(unlabeled[i])
        del unlabeled[i]

    # Save updated session state
    request.session['labeled'] = labeled
    request.session['unlabeled'] = unlabeled

    session_data = {
        'X_all': X_all,
        'y_all': y_all,
        'X_test': X_test,
        'y_test': y_test,
        'labeled': labeled,
        'unlabeled': unlabeled,
        'accuracy_history': accuracy_history,
        'strategy': strategy
    }
    with open(os.path.join(settings.BASE_DIR, 'static', 'active_session.pkl'), 'wb') as f:
        pickle.dump(session_data, f)

    # Plot accuracy over iterations
    plt.figure()
    plt.plot(range(1, len(accuracy_history)+1), accuracy_history, marker='o')
    plt.title("Accuracy Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plot_path = os.path.join(settings.BASE_DIR, 'static', 'accuracy_plot.png')
    plt.savefig(plot_path)
    plt.close()

    return render(request, 'project2/active_loop.html', {
        'accuracy': f"{accuracy:.2f}%",
        'labeled_count': len(labeled),
        'remaining': len(unlabeled),
        'accuracy_plot': 'accuracy_plot.png',
    })

def reset_active_learning(request):
    if request.method == "POST":
        # Delete session variables
        keys_to_clear = ['X_all', 'y_all', 'X_test', 'y_test',
                         'labeled', 'unlabeled', 'strategy', 'accuracy_history']
        for key in keys_to_clear:
            request.session.pop(key, None)

        # Delete saved pickle session
        session_file = os.path.join(settings.BASE_DIR, 'static', 'active_session.pkl')
        if os.path.exists(session_file):
            os.remove(session_file)

        # Load default dataset again and re-initialize with 10 labeled samples
        try:
            df = pd.read_csv(os.path.join(settings.BASE_DIR, 'static', 'IMDB Dataset.csv')).sample(2000, random_state=42)
            df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
            reviews = df['review'].tolist()
            labels = df['label'].tolist()

            X_train_val, X_test, y_train_val, y_test = train_test_split(
                reviews, labels, test_size=0.2, random_state=42
            )

            request.session['X_all'] = X_train_val
            request.session['y_all'] = y_train_val
            request.session['X_test'] = X_test
            request.session['y_test'] = y_test
            request.session['labeled'] = list(range(10))
            request.session['unlabeled'] = list(range(10, len(X_train_val)))
            request.session['strategy'] = 'uncertainty'
            request.session['accuracy_history'] = []

            return redirect('project2:active_learning_loop')

        except Exception as e:
            return render(request, 'project2/active_upload.html', {
                'error': f"Reset failed: {str(e)}"
            })

    return redirect('project2:active_learning')

def reset_active_session(request):
    session_file = os.path.join(settings.BASE_DIR, 'static', 'active_session.pkl')
    if os.path.exists(session_file):
        os.remove(session_file)
    return redirect('project2:active_learning')