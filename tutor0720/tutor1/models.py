from django.db import models
from unicodedata import name
import tutor1
# Create your models here.

class User( models.Model ):
    id = models.AutoField( primary_key = True )
    userID = models.IntegerField() # accepted integer data only.
    age = models.IntegerField() # accepted character data only.
    
class Movie( models.Model ):
    id = models.AutoField( primary_key = True )
    name = models.CharField( max_length = 50 )
    published_year = models.IntegerField()
    poster_dir = models.CharField( max_length = 200 )

class Rating( models.Model ):
    userID = models.ForeignKey( User, on_delete = models.CASCADE )
    movie = models.ForeignKey( Movie, on_delete = models.CASCADE )
    rating = models.IntegerField()
