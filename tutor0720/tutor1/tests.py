from django.test import TestCase

# Create your tests here.
str1 = "123"
str2 = "a123"

def transform_dtype( str_input ):
    try:
        ret = int( str_input )
    except ValueError:
        print( "invalid input type" )
        ret = None
    return ret

print( str1, transform_dtype( str1 ) )
print( str2, transform_dtype( str2 ) )