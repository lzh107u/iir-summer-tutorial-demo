from django.shortcuts import render
from django.http import HttpResponse
from django import forms
from django.template.loader import render_to_string
from tutor1.models import Movie
from tutor1.models import User
from tutor1.models import Rating

from tutor1 import similarity
# base directory for importing is '/tutor0720', 
# which is the same base directory to manage.py
from tutor1 import nlp_app
import pickle
import os
template_base_route = 'templates/tutor1/'

# Create your views here.
def index( request ):
    return HttpResponse( "Hello, world!!" )

def get_something( request, name ):
    return HttpResponse( "return something:" + name )

def show_database( request, table_name = 'default' ):
    print( 'show_database called, table_name:', table_name )

    if MOVIE_INIT == False:
        database_init_checker()
        
    if ( Movie.objects.filter( name = table_name ).count() == 0 ):
        print( 'movie', table_name, 'does not exist in database.' )
    else:
        movie_obj = Movie.objects.get( name = table_name )
        rating_qs = Rating.objects.filter( movie = movie_obj )
        context = {
            "object_list": rating_qs,
            "title": table_name,
            "count": rating_qs.count(),
        }
        html_string = render_to_string( "tutor1/rating.html", context = context )
        return HttpResponse( html_string )
    # myUser = User.objects.filter( userID = 6188 )
    # print( 'myUser:', myUser )
    # Do not use Table.objects.get():
    # => It will result in an error if the data doesn't exist.
    # => Use Table.objects.filter() instead.
    #   => It can return an empty QuerySet

    
    # python( django ) call html: <app>/template/<app>/ ... 
    # html call html( or css ): <app>/static/<app>/ ... 
    return HttpResponse( 'no such movie in database.' )


def database_clean_up( request, table_name = None ):
    if table_name is None:
        print( 'database_clean_up: table_name is None' )
        return HttpResponse( 'no such movie name.' )
    else:
        print( 'database_clean_up: table:', table_name )
    
    movie_obj = Movie.objects.get( name = table_name )
    rating_obj = Rating.objects.filter( movie = movie_obj )
    rating_obj.delete()
    
    print( 'number of rating record:', rating_obj.count() )

    return HttpResponse( 'database_clean_up() called. Every rating of this movie is removed.' )


def dtype_test( request, var ):
    print( 'views.py - dtype_test, var:', var )
    print( 'views.py - dtype_test, dtype:', type( var ) )
    
    return HttpResponse('current data type:' + str( var ) )

# about render():
# => ref: https://vegibit.com/django-render-function/
# => base route: <proj_name>/template
#       => need to add <app_name> to identify the exact directory.
MOVIE_INIT = False
DEFAULT_MOVIES = [ 'inception', 'thedarkknight', 'interstellar', 'dunkirk', 'tenet', 'oppenheimer' ]

def arg_precheck( argument, arg_dtype = None ):
    if len( argument ) == 0:
        # if nothing is in the argument input string, then the entire input is invalid.
        return False
    else:
        if arg_dtype is None:
            # if no dtype is specified, then the pre-checking process ends here.
            return True
        elif type( arg_dtype ) is type( int() ):
            try:
                ret = int( argument )
                return True
            except ValueError:
                print( "view, arg_precheck: invalid input type" )
                return False
        else:
            print( "view, arg_precheck: dtype", type( arg_dtype ), "is currently not supported." )
            return True

MovieRatingArgName = [ 'UserID', 'Age', 'MovieName', 'Score' ]
MovieRatingArgDtype = [ int(), int(), None, int() ]

def request_analysis( request ):
    global MOVIE_INIT
    global MovieRatingArgName
    global MovieRatingArgDtype

    if request.method == "POST":
        form_type = request.POST.get( 'FormType' ) 
        # use <input type="hidden" value="you_need"> to send fixed hidden data.
        # further ref: https://stackoverflow.com/questions/50497871/django-get-id-of-posted-form-in-post-method-of-view-that-contains-list-of-form
        if form_type == 'MovieRatingForm':
            userID = request.POST.get( 'UserID' )
            age = request.POST.get( 'Age' )
            movie = request.POST.get( 'MovieName' )
            score = request.POST.get( 'Score' )

            for index, arg_name in enumerate( MovieRatingArgName ):
                arg = request.POST.get( arg_name )
                print( 'view, request_analysis: index', index, 'arg_name', arg_name, 'arg', arg )
                if ( arg_precheck( arg, MovieRatingArgDtype[ index ] ) == False ):
                    # if any argument is invalid, the entire input sequence is cancelled.
                    print( 'view, request_analysis: input', index, 'is invalid.' )
                    return render( request, 'tutor1/index_v2.html' )
        
            database_input_handler( userID, age, movie, score )
        elif form_type == 'NLPForm':
            sample_text = request.POST.get( 'InputSentence' )
            
            answer = nlp_app.main_task( sample_text = sample_text )
            print( 'request_analysis, nlp prediction answer:', answer )

    return render( request, 'tutor1/index_v2.html' )

def load_params():
    work_dir = os.getcwd()
    print( 'load_params:', work_dir )
    nlp_params_dict = 3
    
    with open( work_dir + '/tutor1/nlp.pickle', 'r' ) as f:
        print('read a file')
        
    
    return nlp_params_dict

def database_init_checker():
    global DEFAULT_MOVIES
    global MOVIE_INIT
    for movie in DEFAULT_MOVIES:
        if ( Movie.objects.filter( name = movie ).count() == 0 ):
            poster_name = "poster_" + movie + '.jpg'
            new_movie = Movie( name = movie, published_year = 2022, poster_dir = "Album_example_Bootstrap_v5_2_files/" + poster_name )
            new_movie.save()

    MOVIE_INIT = True
    return

def database_input_handler( userID, age, movie, score ):
    
    if ( Movie.objects.filter( name = movie ).count() == 0 ):
        # if the input movie has not existed in database yet.
        poster_name = "poster_" + movie + '.jpg'
        new_movie = Movie( name = movie, published_year = 2022, poster_dir = "Album_example_Bootstrap_v5_2_files/" + poster_name )
        new_movie.save()
    
    if ( User.objects.filter( userID = int( userID ) ).count() == 0 ):
        # if the input userID has not existed in database yet.
        new_user = User( userID = int( userID ), age = int( age ) )
        new_user.save()
    
    movie_obj = Movie.objects.get( name = movie )
    user_obj = User.objects.get( userID = int( userID ) )

    if ( Rating.objects.filter( userID = user_obj, movie = movie_obj ).count() == 0 ):
        # if the user has not rated the movie yet.
        new_rating = Rating( userID = user_obj, movie = movie_obj, rating = int( score ) )
        new_rating.save()
    else: 
        alt_rating = Rating.objects.get( userID = user_obj, movie = movie_obj )
        alt_rating.rating = int( score )
        alt_rating.save()

    return

def movie_page_render( request ):
    movies_page_string = render_to_string( "tutor1/Movie_Rating_Page.html" )
    return HttpResponse( movies_page_string )

def analysis_similarity( request, index = 1 ):
    img, match_pair = similarity.main_proc( index = index )
    pic_pairs = []

    if index == 10:
        next_index = 1
    else:
        next_index = index + 1
    print( 'view, analysis_similarity: next_index: ', next_index )

    for count, pair in enumerate( match_pair ):
        # pic_name = "{% static \'" + pair[ 0 ] + "\'%}"
        pic_name = pair[ 0 ] # './../../static/tutor1/template_matching/' + 
        print( pic_name )
        legend = 'best match: ' + str( pair[ 1 ] )
        index = 'template' + str( count )
        template = {
            "pic_name" : pic_name,
            "legend" : legend,
            "index" : index,
        }
        pic_pairs.append( template )

    context = {
        "match_list" : pic_pairs,
        "next" : str( next_index ),
    }
    
    html_string = render_to_string( "tutor1/similarity.html", context = context )
    default_output = 'nothing here.'
    return HttpResponse( html_string )

