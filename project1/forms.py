from django import forms

class CSVUploadForm(forms.Form):
    file = forms.FileField(label='Select a CSV file', help_text='Please upload a CSV file') 

class ModelTrainForm(forms.Form):
    MODEL_CHOICES = [
        ('rf', 'Random Forest'),
        ('svm', 'Support Vector Machine'),
        ('xgb', 'Extreme Gradient Boost'),
    ]
    
    model_type = forms.ChoiceField(choices=MODEL_CHOICES, label="Model Type")
    test_size = forms.FloatField(min_value=0.1, max_value=0.9, initial=0.2, label="Test Size (0.1 to 0.9)")
    
    # Random Forest parameters
    rf_n_estimators = forms.IntegerField(min_value=10, max_value=500, initial=100, 
                                       label="Number of Estimators", required=False,
                                       help_text="Number of trees in the forest")
    rf_max_depth = forms.IntegerField(min_value=1, max_value=50, initial=5, 
                                     label="Max Depth", required=False,
                                     help_text="Maximum depth of the trees")
    RF_MAX_FEATURES_CHOICES = [
        ('sqrt', 'Square Root'),
        ('log2', 'Log2'),
        ('None', 'All Features'),
    ]
    rf_max_features = forms.ChoiceField(choices=RF_MAX_FEATURES_CHOICES, 
                                       initial='sqrt', 
                                       label="Max Features", 
                                       required=False,
                                       help_text="Number of features to consider for best split")
    
    # SVM parameters
    SVM_KERNEL_CHOICES = [
        ('rbf', 'RBF'),
        ('linear', 'Linear'),
        ('poly', 'Polynomial'),
        ('sigmoid', 'Sigmoid'),
    ]
    svm_kernel = forms.ChoiceField(choices=SVM_KERNEL_CHOICES, 
                                  initial='rbf', 
                                  label="Kernel", 
                                  required=False,
                                  help_text="Kernel type to be used in the algorithm")
    svm_C = forms.FloatField(min_value=0.1, max_value=10.0, initial=1.0, 
                           label="C (Regularization)", required=False,
                           help_text="Regularization parameter")
    svm_gamma = forms.FloatField(min_value=0.001, max_value=1.0, initial=0.1, 
                               label="Gamma", required=False,
                               help_text="Kernel coefficient")
    
    # XGBoost parameters
    xgb_n_estimators = forms.IntegerField(min_value=10, max_value=500, initial=100, 
                                        label="Number of Estimators", required=False,
                                        help_text="Number of boosting rounds")
    xgb_learning_rate = forms.FloatField(min_value=0.001, max_value=1.0, initial=0.01, 
                                       label="Learning Rate", required=False,
                                       help_text="Step size shrinkage used to prevent overfitting")
    xgb_max_depth = forms.IntegerField(min_value=1, max_value=15, initial=3, 
                                     label="Max Depth", required=False,
                                     help_text="Maximum depth of a tree")
