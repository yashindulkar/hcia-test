from .utils import simulate_rating_impact
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import io
from reportlab.pdfgen import canvas
from pathlib import Path
import pandas as pd
from .models import Feedback 
import logging
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime
    

logger = logging.getLogger(__name__)

def project4_landing(request):
    """
    This is the landing page for Project 4.
    It includes a link to download the PDF and a button to start the study.
    """
    return render(request, 'project4/landing.html')


def project4_study(request):
    BASE_DIR = Path(__file__).resolve().parent
    movies_path = BASE_DIR / "data" / "movies.csv"
    
    try:
        movies_df = pd.read_csv(movies_path)
        movie_title_to_id = dict(zip(movies_df['title'], movies_df['movieId']))
        all_titles = sorted(movie_title_to_id.keys())
        logger.info(f"Loaded {len(all_titles)} movies from dataset")
    except Exception as e:
        logger.error(f"Error loading movies: {str(e)}")
        # Fallback movie list for demo
        all_titles = [
            "Toy Story (1995)",
            "Jumanji (1995)", 
            "Grumpier Old Men (1995)",
            "Waiting to Exhale (1995)",
            "Father of the Bride Part II (1995)"
        ]
        movie_title_to_id = {title: i+1 for i, title in enumerate(all_titles)}

    selected_movie = None
    prediction_impact = None
    feedback_submitted = False
    error_message = None

    if request.method == "POST":
        selected_movie = request.POST.get("movie_title", "").strip()
        logger.info(f"User selected movie: '{selected_movie}'")
        
        if 'submit_feedback' in request.POST:
            # Save feedback
            try:
                Feedback.objects.create(
                    movie_title=selected_movie if selected_movie else "Unknown",
                    helpfulness=request.POST.get("feedback", "None"),
                    comments=request.POST.get("comments", "")
                )
                feedback_submitted = True
                logger.info(f"Feedback saved for movie: {selected_movie}")
            except Exception as e:
                logger.error(f"Error saving feedback: {str(e)}")
                error_message = "Error saving feedback. Please try again."

        # Calculate prediction impact if a movie is selected
        if selected_movie and selected_movie in movie_title_to_id:
            try:
                logger.info(f"Calculating prediction impact for: {selected_movie}")
                prediction_impact = simulate_rating_impact(selected_movie, movie_title_to_id)
                
                # Debug: log the results
                for rating, movies in prediction_impact.items():
                    logger.debug(f"{rating}: {len(movies)} movies - {movies}")
                    
                # Validate that we got different results
                if prediction_impact:
                    all_recommendations = []
                    for movies in prediction_impact.values():
                        all_recommendations.extend(movies)
                    
                    unique_recommendations = len(set(all_recommendations))
                    total_recommendations = len(all_recommendations)
                    
                    logger.info(f"Generated {unique_recommendations} unique recommendations out of {total_recommendations} total")
                    
                    # Check if results seem too similar (potential issue)
                    if unique_recommendations < total_recommendations * 0.7:
                        logger.warning("Recommendations seem too similar - check model diversity")
                        
            except Exception as e:
                logger.error(f"Error calculating prediction impact: {str(e)}")
                error_message = f"Error calculating recommendations for '{selected_movie}'. Please try a different movie."
                prediction_impact = None
        elif selected_movie and selected_movie not in movie_title_to_id:
            error_message = f"Movie '{selected_movie}' not found in our database."
            logger.warning(f"Movie not found: {selected_movie}")

    # Add some analytics data for the other tabs
    try:
        recent_feedback = Feedback.objects.all().order_by('-submitted_at')[:10]
        total_submissions = Feedback.objects.count()
        
        # Calculate some basic stats
        feedback_stats = {
            'total_submissions': total_submissions,
            'recent_feedback': recent_feedback,
        }
    except Exception as e:
        logger.error(f"Error fetching analytics: {str(e)}")
        feedback_stats = {
            'total_submissions': 0,
            'recent_feedback': [],
        }

    context = {
        'movie_title': selected_movie,
        'prediction_impact': prediction_impact,
        'all_titles': all_titles,
        'feedback_submitted': feedback_submitted,
        'error_message': error_message,
        **feedback_stats
    }
    
    return render(request, 'project4/study.html', context)


# Add a debug endpoint to test the prediction system
def debug_predictions(request):
    """Debug endpoint to test prediction generation"""
    if request.method == 'GET':
        movie_title = request.GET.get('movie', 'Toy Story (1995)')
        
        BASE_DIR = Path(__file__).resolve().parent
        movies_path = BASE_DIR / "data" / "movies.csv"
        
        try:
            movies_df = pd.read_csv(movies_path)
            movie_title_to_id = dict(zip(movies_df['title'], movies_df['movieId']))
        except:
            movie_title_to_id = {"Toy Story (1995)": 1}
        
        predictions = simulate_rating_impact(movie_title, movie_title_to_id)
        
        return JsonResponse({
            'movie': movie_title,
            'predictions': predictions,
            'debug_info': {
                'total_movies_in_db': len(movie_title_to_id),
                'movie_found': movie_title in movie_title_to_id,
                'movie_id': movie_title_to_id.get(movie_title)
            }
        })


def project4_download_pdf(request):
    """
    Generate comprehensive PDF with Task 1 and Task 2 content
    """
   
    buffer = io.BytesIO()
    
    # Create PDF document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Get styles and create custom styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkblue
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=16,
        textColor=colors.darkred
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        leading=14
    )
    
    # Content elements
    story = []
    
    # Title Page
    story.append(Paragraph("Project 4: Influence of Future Predictions over Active Learning of Users' Tastes for Recommender Systems", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Human-Centric Artificial Intelligence", styles['Heading2']))
    story.append(Spacer(1, 30))
    
    # Date
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Abstract
    abstract_text = """
    <b>Abstract:</b> This document presents the methodology and user study design for investigating 
    how showing users the impact of their responses affects the learning process in recommendation 
    systems. We explore a guided active learning approach for cold-start recommendation that provides 
    users with information about how their ratings will influence future recommendations.
    """
    story.append(Paragraph(abstract_text, body_style))
    story.append(PageBreak())
    
    # Table of Contents
    story.append(Paragraph("Table of Contents", heading_style))
    toc_data = [
        ["1. Task 1: Guided Active Learning Method", "3"],
        ["   1.1 Problem Definition", "3"],
        ["   1.2 Methodology", "3"],
        ["   1.3 Matrix Factorization Approach", "4"],
        ["   1.4 Impact Visualization", "5"],
        ["2. Task 2: User Study Design", "6"],
        ["   2.1 Research Hypothesis", "6"],
        ["   2.2 Study Design", "7"],
        ["   2.3 Participants and Recruitment", "7"],
        ["   2.4 Experimental Procedure", "8"],
        ["   2.5 Evaluation Metrics", "9"],
        ["3. Implementation Details", "10"],
        ["4. Expected Outcomes", "11"]
    ]
    
    toc_table = Table(toc_data, colWidths=[4*inch, 0.5*inch])
    toc_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(toc_table)
    story.append(PageBreak())
    
    # TASK 1: METHODOLOGY
    story.append(Paragraph("1. Task 1: Guided Active Learning Method", heading_style))
    
    # 1.1 Problem Definition
    story.append(Paragraph("1.1 Problem Definition", subheading_style))
    problem_text = """
    Cold start is a fundamental challenge in recommender systems. When new users join a platform, 
    the system lacks information about their preferences, making it difficult to provide personalized 
    recommendations. Traditional active learning approaches ask users to rate items without providing 
    context about how these ratings will influence their future experience.
    
    Our approach addresses this limitation by implementing a transparent recommendation process that 
    shows users how their ratings will impact future recommendations, potentially leading to more 
    thoughtful and strategic rating decisions.
    """
    story.append(Paragraph(problem_text, body_style))
    
    # 1.2 Methodology
    story.append(Paragraph("1.2 Methodology", subheading_style))
    methodology_text = """
    <b>Core Innovation:</b> Unlike standard active learning, our method provides users with an 
    indication of how their answer will affect the system. This transparency allows users to 
    visualize the effect of their replies and decide to answer more strategically.
    
    <b>Key Components:</b>
    <br/>• Interactive rating interface with real-time impact visualization
    <br/>• Matrix factorization-based recommendation engine
    <br/>• Impact simulation showing recommendations for different rating scenarios (1, 3, and 5 stars)
    <br/>• User feedback collection system for evaluating method effectiveness
    """
    story.append(Paragraph(methodology_text, body_style))
    
    # 1.3 Matrix Factorization
    story.append(Paragraph("1.3 Matrix Factorization Approach", subheading_style))
    matrix_text = """
    We employ matrix factorization as our core recommendation algorithm. The approach minimizes:
    
    <br/><br/><b>Objective Function:</b>
    <br/>min<sub>U,V</sub> Σ<sub>(i,j)∈K</sub> ||R<sub>ij</sub> - U<sub>i</sub><sup>T</sup>V<sub>j</sub>||² + λ(||U||²<sub>F</sub> + ||V||²<sub>F</sub>)
    
    <br/><br/>Where:
    <br/>• R<sub>ij</sub> is the rating user i gave to item j
    <br/>• U<sub>i</sub> ∈ R<sup>K</sup> represents user i in K-dimensional latent space
    <br/>• V<sub>j</sub> represents item j in the same latent space
    <br/>• λ is the regularization parameter
    
    <br/><br/>For new users, we learn their representation by solving:
    <br/>min<sub>U<sub>i</sub></sub> Σ<sub>j∈K<sub>i</sub></sub> ||R<sub>ij</sub> - U<sub>i</sub><sup>T</sup>V<sub>j</sub>||² + λ||U<sub>i</sub>||²<sub>F</sub>
    """
    story.append(Paragraph(matrix_text, body_style))
    
    # 1.4 Impact Visualization
    story.append(Paragraph("1.4 Impact Visualization", subheading_style))
    visualization_text = """
    <b>Rating Impact Simulation:</b>
    For each selected movie, the system simulates three scenarios:
    
    <br/>• <b>1-Star Rating:</b> Simulates user dislike, generates recommendations for users with 
    contrasting preferences
    <br/>• <b>3-Star Rating:</b> Simulates neutral preference, provides balanced recommendations
    <br/>• <b>5-Star Rating:</b> Simulates strong preference, suggests similar high-quality content
    
    <br/><br/><b>Visualization Components:</b>
    <br/>• Bar chart showing number of recommendations per rating scenario
    <br/>• Detailed recommendation lists for each rating level
    <br/>• Real-time updates as users select different movies
    
    <br/><br/>This transparency allows users to understand the consequences of their rating decisions 
    and potentially provide more accurate preference information.
    """
    story.append(Paragraph(visualization_text, body_style))
    story.append(PageBreak())
    
    # TASK 2: USER STUDY DESIGN
    story.append(Paragraph("2. Task 2: User Study Design", heading_style))
    
    # 2.1 Research Hypothesis
    story.append(Paragraph("2.1 Research Hypothesis", subheading_style))
    hypothesis_text = """
    <b>Primary Hypothesis:</b> Showing users the impact of their rating decisions on future 
    recommendations will lead to more thoughtful rating behavior and improved recommendation quality.
    
    <b>Secondary Hypotheses:</b>
    <br/>• Users will find the impact visualization helpful for understanding the recommendation system
    <br/>• Transparency will increase user trust and satisfaction with the recommendation process
    <br/>• Users will be more likely to provide accurate ratings when they understand the consequences
    <br/>• The approach will be particularly effective for users with strong genre preferences
    """
    story.append(Paragraph(hypothesis_text, body_style))
    
    # 2.2 Study Design
    story.append(Paragraph("2.2 Study Design", subheading_style))
    study_design_text = """
    <b>Study Type:</b> Within-subjects experimental design with user experience evaluation
    
    <b>Independent Variables:</b>
    <br/>• Movie selection (participants choose from 9,742 available movies)
    <br/>• Rating visualization presence (our system shows impact prediction)
    
    <b>Dependent Variables:</b>
    <br/>• User satisfaction with recommendation transparency (5-point Likert scale)
    <br/>• Perceived helpfulness of impact visualization
    <br/>• Time spent considering rating decisions
    <br/>• Qualitative feedback on user experience
    
    <b>Control Measures:</b>
    <br/>• All participants use the same movie database (MovieLens 100k dataset)
    <br/>• Consistent interface design and interaction patterns
    <br/>• Standardized instructions and task presentation
    """
    story.append(Paragraph(study_design_text, body_style))
    
    # 2.3 Participants and Recruitment
    story.append(Paragraph("2.3 Participants and Recruitment", subheading_style))
    participants_text = """
    <b>Target Sample Size:</b> 50-100 participants
    
    <b>Inclusion Criteria:</b>
    <br/>• Age 18 or older
    <br/>• Regular movie/streaming service users (at least 2 movies per month)
    <br/>• Basic computer literacy for web interface interaction
    <br/>• English language proficiency for movie title comprehension
    
    <b>Recruitment Strategy:</b>
    <br/>• University student populations (convenient sampling)
    <br/>• Online movie/entertainment forums and communities
    <br/>• Social media recruitment with movie-related hashtags
    <br/>• Snowball sampling through initial participants
    
    <b>Incentives:</b>
    <br/>• Entry into movie ticket gift card lottery
    <br/>• Personalized movie recommendation report
    <br/>• Course credit for student participants (where applicable)
    """
    story.append(Paragraph(participants_text, body_style))
    
    # 2.4 Experimental Procedure
    story.append(Paragraph("2.4 Experimental Procedure", subheading_style))
    procedure_text = """
    <b>Session Duration:</b> 15-20 minutes per participant
    
    <b>Procedure Steps:</b>
    
    <br/><b>1. Introduction (3 minutes)</b>
    <br/>• Welcome and consent process
    <br/>• Brief explanation of recommendation systems
    <br/>• Interface orientation and tutorial
    
    <br/><b>2. Main Task (10-12 minutes)</b>
    <br/>• Participants select 3-5 movies they are familiar with
    <br/>• For each movie, view the rating impact visualization
    <br/>• Provide feedback on helpfulness after each interaction
    <br/>• Complete satisfaction questionnaire
    
    <br/><b>3. Feedback Collection (3-5 minutes)</b>
    <br/>• Post-task interview about experience
    <br/>• Suggestions for system improvement
    <br/>• Demographic information collection
    
    <b>Data Collection Methods:</b>
    <br/>• Automatic logging of user interactions
    <br/>• Likert scale questionnaires
    <br/>• Open-ended feedback comments
    <br/>• Optional brief exit interview
    """
    story.append(Paragraph(procedure_text, body_style))
    
    # 2.5 Evaluation Metrics
    story.append(Paragraph("2.5 Evaluation Metrics", subheading_style))
    metrics_text = """
    <b>Quantitative Metrics:</b>
    
    <br/><b>User Satisfaction:</b>
    <br/>• Overall helpfulness rating (1-5 scale)
    <br/>• System transparency perception
    <br/>• Likelihood to use similar features
    
    <br/><b>Behavioral Metrics:</b>
    <br/>• Time spent viewing impact visualizations
    <br/>• Number of movies explored per session
    <br/>• Interaction patterns with recommendation lists
    
    <br/><b>Qualitative Metrics:</b>
    <br/>• Thematic analysis of user comments
    <br/>• Categorization of feature improvement suggestions
    <br/>• User mental model assessment of recommendation process
    
    <b>Success Criteria:</b>
    <br/>• >70% of users rate the system as helpful (4+ on 5-point scale)
    <br/>• Positive qualitative feedback themes outweigh negative
    <br/>• Users demonstrate understanding of rating impact concept
    """
    story.append(Paragraph(metrics_text, body_style))
    story.append(PageBreak())
    
    # Implementation Details
    story.append(Paragraph("3. Implementation Details", heading_style))
    implementation_text = """
    <b>Technical Architecture:</b>
    
    <br/><b>Backend:</b>
    <br/>• Django web framework for server-side logic
    <br/>• Python-based matrix factorization implementation
    <br/>• SQLite database for user feedback storage
    <br/>• MovieLens 100k dataset for movie recommendations
    
    <br/><b>Frontend:</b>
    <br/>• HTML5/CSS3/JavaScript for responsive interface
    <br/>• Chart.js for rating impact visualizations
    <br/>• Bootstrap framework for mobile compatibility
    <br/>• AJAX for real-time recommendation updates
    
    <br/><b>Data Pipeline:</b>
    <br/>• Real-time prediction simulation using pre-trained models
    <br/>• Caching layer for improved response times
    <br/>• Fallback recommendation system for robustness
    <br/>• Comprehensive logging for user behavior analysis
    
    <br/><b>User Interface Features:</b>
    <br/>• Movie search and selection interface
    <br/>• Interactive rating impact charts
    <br/>• Detailed recommendation breakdowns
    <br/>• Feedback collection forms
    <br/>• Multi-tab analytics dashboard for researchers
    """
    story.append(Paragraph(implementation_text, body_style))
    
    # Expected Outcomes
    story.append(Paragraph("4. Expected Outcomes", heading_style))
    outcomes_text = """
    <b>Research Contributions:</b>
    
    <br/><b>Theoretical:</b>
    <br/>• Enhanced understanding of transparency effects in recommender systems
    <br/>• Insights into user decision-making when rating consequences are visible
    <br/>• Framework for guided active learning in cold-start scenarios
    
    <br/><b>Practical:</b>
    <br/>• Prototype system demonstrating transparent recommendation process
    <br/>• User experience guidelines for recommendation system interfaces
    <br/>• Evaluation methodology for active learning approaches
    
    <b>Potential Impact:</b>
    <br/>• Improved user onboarding for streaming and e-commerce platforms
    <br/>• Enhanced user trust through algorithmic transparency
    <br/>• Better quality training data through informed user feedback
    <br/>• Reduced cold-start problem effects in real-world systems
    
    <b>Future Research Directions:</b>
    <br/>• Extension to other domains beyond movie recommendations
    <br/>• Investigation of long-term effects on user behavior
    <br/>• Comparison with other active learning approaches
    <br/>• Integration with advanced machine learning models
    """
    story.append(Paragraph(outcomes_text, body_style))
    story.append(PageBreak())
    
    # References section
    story.append(Spacer(1, 20))
    story.append(Paragraph("References", heading_style))
    references_text = """
    <br/>• Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. 
    ACM Transactions on Interactive Intelligent Systems, 5(4), 1-19.
    
    <br/>• Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for 
    recommender systems. Computer, 42(8), 30-37.
    
    <br/>• Settles, B. (2009). Active learning literature survey. University of Wisconsin-Madison 
    Department of Computer Sciences.
    
    <br/>• Knijnenburg, B. P., Willemsen, M. C., Gantner, Z., Soncu, H., & Newell, C. (2012). 
    Explaining the user experience of recommender systems. User Modeling and User-Adapted 
    Interaction, 22(4-5), 441-504.
    """
    story.append(Paragraph(references_text, body_style))
    
    # Build PDF
    doc.build(story)
    
    # Return response
    buffer.seek(0)
    response = HttpResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="Project4_Method_UserStudy_{datetime.now().strftime("%Y%m%d")}.pdf"'
    
    return response

def project4_feedback_review(request):
    feedbacks = Feedback.objects.all().order_by('-id')
    return render(request, 'project4/feedback_review.html', {
        'feedbacks': feedbacks,
        'page_title': "Feedback Review"
    })


def project4_export_feedback(request):
    """
    Export feedback data as CSV
    """
    import csv
    from django.http import HttpResponse
    from datetime import datetime
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="feedback_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['ID', 'Movie Title', 'Helpfulness', 'Comments', 'Submitted At'])
    
    feedbacks = Feedback.objects.all().order_by('-submitted_at')
    for feedback in feedbacks:
        writer.writerow([
            feedback.id,
            feedback.movie_title,
            feedback.helpfulness,
            feedback.comments,
            feedback.submitted_at.strftime('%Y-%m-%d %H:%M:%S')
        ])
    
    return response


def project4_analytics(request):
    """
    Analytics dashboard for Project 4 - Movie Recommender Study
    """
    try:
        # Get all feedback data
        feedbacks = Feedback.objects.all()
        total_submissions = feedbacks.count()
        
        # Calculate helpfulness statistics
        helpfulness_counts = {}
        for feedback in feedbacks:
            helpfulness = feedback.helpfulness
            helpfulness_counts[helpfulness] = helpfulness_counts.get(helpfulness, 0) + 1
        
        # Most popular movies (movies that were tested most)
        movie_counts = {}
        for feedback in feedbacks:
            movie = feedback.movie_title
            movie_counts[movie] = movie_counts.get(movie, 0) + 1
        
        popular_movies = sorted(movie_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Recent activity
        recent_feedback = feedbacks.order_by('-submitted_at')[:10]
        
        context = {
            'page_title': 'Project 4 Analytics',
            'total_submissions': total_submissions,
            'helpfulness_counts': helpfulness_counts,
            'popular_movies': popular_movies,
            'recent_feedback': recent_feedback,
        }
        
    except Exception as e:
        # Fallback context if there are any issues (e.g., no database entries yet)
        context = {
            'page_title': 'Project 4 Analytics',
            'total_submissions': 0,
            'helpfulness_counts': {},
            'popular_movies': [],
            'recent_feedback': [],
            'error': str(e)
        }
    
    return render(request, 'project4/analytics.html', context)