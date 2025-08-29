# Quick Setup Guide for Instructor

## One-Command Setup

```bash
# Clone repository
git clone <repository-url>
cd HCAI-PBL

# Setup environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# Install all dependencies
pip install -r requirements.txt

# Setup database
python manage.py makemigrations
python manage.py migrate

# Run server
python manage.py runserver
```

**Access**: http://127.0.0.1:8000/home/

## Required Datasets

### Automatic (No Setup Needed)
- **Project 1**: Uses uploaded CSV files
- **Project 3**: Palmer Penguins (auto-downloaded)
- **Project 4**: MovieLens data included in repo
- **Project 5**: Custom grid environment (included)

### Manual Setup Required
**Project 2 Only**: IMDB 50k Reviews
- Download: https://ai.stanford.edu/~amaas/data/sentiment/
- Extract to: `project2/data/imdb_reviews/`

## Project Access URLs
- **Home/Navigation**: `/home/`
- **Project 1 - AutoML**: `/project1/`
- **Project 2 - Active Learning**: `/project2/`
- **Project 3 - Explainability**: `/project3/`
- **Project 4 - Recommenders**: `/project4/`
- **Project 5 - RL+HF**: `/project5/`

## System Requirements
- Python 3.8+
- 4GB RAM minimum
- Modern web browser

## Troubleshooting
```bash
# If imports fail
pip install --upgrade pip
pip install -r requirements.txt

# If database errors
rm db.sqlite3
python manage.py migrate

# If static files missing
python manage.py collectstatic --noinput
```
