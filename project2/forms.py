from django import forms

class TextClassificationForm(forms.Form):
    file = forms.FileField(label="Upload CSV")

    vectorizer_choice = forms.ChoiceField(
        choices=[
            ('tfidf', 'TF-IDF'),
            ('bow', 'Bag of Words')
        ],
        label="Text Representation"
    )

    model_choice = forms.ChoiceField(
        choices=[
            ('logreg', 'Logistic Regression'),
            ('svm', 'Support Vector Machine'),
            ('rf', 'Random Forest')
        ],
        label="Classifier"
    )