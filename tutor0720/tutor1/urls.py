from django.urls import path
from tutor1 import views
from django.conf import settings
from django.conf.urls.static import static
# trouble fixed
# Warning: the url list should be exactly named 'urlpatterns'.
# An error will occurred if it's misspelling.
urlpatterns = [
    path( 'home', views.index, name = 'index' ),
    path( 'name/<str:name>/', views.get_something, name = 'get_something' ),
    path( 'vartest/<int:var>/', views.dtype_test, name = 'dtype_test' ),
    
    path( 'analysis/', views.request_analysis ),
    path( 'analysis/query/<str:table_name>/', views.show_database, name = 'show_database' ),
    path( 'analysis/clean/<str:table_name>/', views.database_clean_up, name = 'database_clean_up' ),
    path( 'analysis/similarity/', views.analysis_similarity, name = 'analysis_similarity' ),
    path( 'analysis/similarity/<int:index>/', views.analysis_similarity, name = 'analysis_similarity' ),
    path( 'analysis/movies/', views.movie_page_render, name = "movie_page_render" ),
    
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

