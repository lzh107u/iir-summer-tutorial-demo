import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

"""
Issue: cvNamedWindow is not implemented.
    => https://stackoverflow.com/questions/67120450/error-2unspecified-error-the-function-is-not-implemented-rebuild-the-libra
    
        pip3 uninstall opencv-python-headless -y
        pip3 install opencv-python --upgrade
"""

top_dir = None
mnist = []
area_lower_bound = 60
margin_stripe = 3
gradient_template = [
    [ [ 0, 0, 0, 0 ], [ 0, 0, 0, 0 ], [ 1, 1, 1, 1 ], [ 1, 1, 1, 1 ] ], # horizontal stripe
    [ [ 0, 0, 1, 1 ], [ 0, 0, 1, 1 ], [ 0, 0, 1, 1 ], [ 0, 0, 1, 1 ] ], # vertical stripe
    [ [ 1, 0, 0, 0 ], [ 1, 1, 0, 0 ], [ 1, 1, 1, 0 ], [ 1, 1, 1, 1 ] ], # top-left to bottom-right
    [ [ 0, 0, 0, 1 ], [ 0, 0, 1, 1 ], [ 0, 1, 1, 1 ], [ 1, 1, 1, 1 ] ]  # top-right to bottom-left
]
FILENAME_PREFIX = 'match'
FILENAME_POSFIX = '.jpg'
MATCH_PAIR = []
MNIST_DIR = '/tutor1/static/tutor1/mnist/'
DOOR_DIR = '/tutor1/static/tutor1/door/'
SAVE_DIR = '/tutor1/static/tutor1/template_matching/'


def read_door_img( count = 1 ):
    global top_dir
    global DOOR_DIR
    filename = str( count ) + '.png'
    new_dir = top_dir + DOOR_DIR
    os.chdir( new_dir )
    img = cv2.imread( filename )
    os.chdir( top_dir )
    return img

def template_matching( img ):
    global gradient_template
    # winname = 'template matching'
    # cv2.namedWindow( winname, cv2.WINDOW_NORMAL )
    num_row, num_col = img.shape
    img = cv2.resize( img, ( img.shape[ 1 ]*2, img.shape[ 0 ]*4 ) )
    template_vector = []
    
    for row in range( 4 ):
        for col in range( 2 ):
            best_choice = -1
            mae = 255*num_row*num_col
            for index, mask in enumerate( gradient_template ):
                temp = np.ones( ( num_row, num_col ), dtype = np.uint8 ) * 255
                mask = np.array( mask, dtype = np.uint8 )
                mask = cv2.resize( mask, ( temp.shape[ 1 ], temp.shape[ 0 ] ) )
                
                temp = temp*mask
                ret, temp_inv = cv2.threshold( temp, 127, 255, cv2.THRESH_BINARY_INV )
                
                # img_display = img[ row*num_row : ( row + 1 )*num_row, col*num_col : ( col + 1 )*num_col ]
                res_abs = np.abs( temp - img[ row*num_row : ( row + 1 )*num_row, col*num_col : ( col + 1 )*num_col ] )
                ret = np.sum( res_abs )
                if ret < mae:
                    mae = ret
                    best_choice = index
                res_abs = np.abs( temp_inv - img[ row*num_row : ( row + 1 )*num_row, col*num_col : ( col + 1 )*num_col ] )
                
                ret = np.sum( res_abs )
                if ret < mae:
                    mae = ret
                    # invert result
                    best_choice = index
            template_vector.append( best_choice )
    # print( 'template_matching, template_vector:', template_vector )
    # cv2.destroyWindow( winname )
    return template_vector

def template_rebuild( img, vector ):
    global gradient_template
    template = None
    # winname = 'template rebuild'
    # cv2.namedWindow( winname, cv2.WINDOW_NORMAL )
    baseline = np.ones( ( 4, 8 ), dtype = np.uint8 ) * 255
    for row in range( 4 ):
        layer_template = np.concatenate( [ np.array( gradient_template[ vector[ row*2 ] ], dtype = np.uint8 ) ,
                                         np.array( gradient_template[ vector[ row*2 + 1 ] ], dtype = np.uint8 ) ], axis = 1 )
        layer_template = layer_template * baseline
        if template is None:
            template = layer_template    
        else:
            template = np.concatenate( [ template, layer_template ], axis = 0 )
            
    return 

def set_mnist():
    """
    read every mnist template into memory and process it
    in order to be the dictionary for the following input.
    """
    global mnist
    global top_dir
    global gradient_template
    global MNIST_DIR
    # winname = 'mnist demo'
    # cv2.namedWindow( winname, cv2.WINDOW_NORMAL )
    new_dir = top_dir + MNIST_DIR
    os.chdir( new_dir )
    # read in every mnist template
    for i in range( 10 ):
        filename = str( i ) + '.jpg'
        img = cv2.imread( filename )
        # use threshold process to transform the image into 0 and 255 only.
        gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
        ret, thresh = cv2.threshold( gray, thresh = 127, maxval = 255, type = cv2.THRESH_BINARY )
        # find the major contour of each template
        contours, hierarchy = cv2.findContours( thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
        if hierarchy is not None:
            # if the contour exists, then extract it from original image
            x, y, w, h = cv2.boundingRect( contours[ 0 ] )
            thresh = thresh[ y : y + h, x : x + w ] # remove margin

        template_vector = template_matching( thresh )
        template_rebuild( thresh, template_vector )
        mnist.append( ( thresh, template_vector ) )
    
    os.chdir( top_dir )
    
    return 

def partial_enhancement( img_orig, coors ):
    """
    enhance the contrast of contour area on original image.

    return a thresholded contour for recognition.
    or None if ratio of contour is too weird to be a digit.
    """
    # winname = 'partial enhancement'
    # cv2.namedWindow( winname )
    
    # enlarge the original image
    # Attention: the up-scale-parameter need to be the same to the one in single_contour_process()
    img_orig = cv2.resize( img_orig, ( img_orig.shape[ 1 ]*3, img_orig.shape[ 0 ]*3 ) )
    
    # extract ROI
    x, y, w, h = coors
    orig_part = img_orig[ y : y + h, x : x + w ]
    
    # enhance image contrast
    equ = cv2.equalizeHist( orig_part )
    # if the width and height of the equalized img exceeds the given ratio, it will be rejected.
    if equ.shape[ 1 ] * 3 < equ.shape[ 0 ] or equ.shape[ 1 ] > equ.shape[ 0 ]:
        # print( 'reject' )
        return None
    else:
        # with enhanced contrast ROI, the result of threshold will be better 
        # than what it gets before single_contour_process called.
        ret, thresh = cv2.threshold( equ, 100, 255, cv2.THRESH_BINARY )
        
        return thresh # back to sample_comparison()

def single_contour_process( img, img_orig ):
    """
    enlarge the input image for better recognition.

    send the enlarged contour to partial_enhancement for final labeling.
    """
    # winname = 'single contour process'
    # cv2.namedWindow( winname, cv2.WINDOW_NORMAL )

    # enlarge the image
    img = cv2.resize( img, ( img.shape[ 1 ]*3, img.shape[ 0 ]*3 ) )
    
    # perform dilation to the image
    # get region proposal
    d_kernel = np.ones( ( 5, 5 ) )
    dilation = cv2.dilate( img, d_kernel, iterations = 1 ) 
    
    contours, hierarchy = cv2.findContours( dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    
    # find the top contour
    # => the input img contained only one ( possible ) digit
    # => enlarging and dilating can get a clear view of img and simultaneously maintain features.
    # => noise will be produced in this process, so we need to locate the major ( biggest ) contour again.
    for row in range( hierarchy.shape[ 1 ] ):
        if hierarchy[ 0, row, 3 ] == -1:
            # no precedent
            break
        pass
    # the bounding rectangle coors of the major contour
    x, y, w, h = cv2.boundingRect( contours[ row ] )
    
    # Attention: coors are what we want !!
    # => the main goal of this function is to find the coors of digit in a up-scaled img.
    # => at the end, it will pass the coors to partial_enhancement()
    return partial_enhancement( img_orig, ( x, y, w, h ) )

def margin_evaluation( margin ):
    """
    use cv2 built-in histogram to calculate the number of 0 and 255 respectively.

    return:
    => num_255 - num_0
        => if the return value is bigger than 0, we can assume that background is white.
    """
    hist = cv2.calcHist( [ margin ], [ 0 ], None, [ 256 ], [ 0, 256 ] )
    return hist[ 255, 0 ] - hist[ 0, 0 ]

def background_detection( img ):
    """
    decide the input image is white background or black background
    by counting the number of 0-pixel and 255-pixel in the margin part of input image.

    return:
    => 0: black background
    => 1: white background
    """
    global margin_stripe
    if img is None:
        return
    num_row, num_col = img.shape

    ret, thresh = cv2.threshold( img, 127, 255, cv2.THRESH_BINARY )
    margin_type = 0
    margin_type += margin_evaluation( thresh[ 0 : margin_stripe, : ] ) # detect top margin
    margin_type += margin_evaluation( thresh[ num_row - margin_stripe : , : ] ) # detect bottom margin
    margin_type += margin_evaluation( thresh[ : , 0 : margin_stripe ] ) # detect left margin
    margin_type += margin_evaluation( thresh[ : , num_col - margin_stripe ] ) # detect right margin

    if margin_type > 0:
        return 1
    else:
        return 0



def sample_comparison( contours, hierarchy, img, img_orig, pic_index ):
    """
    perform enhancement on each contour to get ROI, 
    and send it to single_contour_process to get a 
    better partial enhanced image from the origin frame.

    next, with the returned enhanced sample, it can search
    the best template in list.
    """
    global area_lower_bound
    global mnist
    global top_dir
    global MATCH_PAIR
    global FILENAME_PREFIX
    global FILENAME_POSFIX
    global SAVE_DIR

    winname = 'contours'
    # cv2.namedWindow( winname, cv2.WINDOW_NORMAL )
    contour_count = hierarchy.shape[ 1 ] # the number of contours on this img

    for count in range( contour_count ):
        contour = contours[ count ]
        img_copy = img.copy()
        img_copy = cv2.cvtColor( img_copy, cv2.COLOR_GRAY2BGR )
        img_res = img_copy.copy()
        
        # draw the contour on the img with green line
        # use '-1' as line parameter => fill the entire contour area with color
        cv2.drawContours( img_copy, [ contour ], 0, ( 0, 255, 0 ), -1 )
        
        # get residual part
        # img_res is the part that surrounded by the contour
        img_res = img_res - img_copy
        
        # the minimum rotated rectangle that contains entire contour
        rect = cv2.minAreaRect( contour )
        # turn contour coordinate into box format
        box = cv2.boxPoints( rect )
        # turn box points into integer type
        box = np.int0( box )
        
        # check if the area is greater than lower bound or not.
        if cv2.contourArea( box ) < area_lower_bound:
            # if the area of a contour box is smaller than the lower bound,
            # there is no need to consider it as a candidate.
            pass
        else:
            # show the contour box and residual part.
            img_res = cv2.cvtColor( img_res, cv2.COLOR_BGR2GRAY )
            ret, img_res = cv2.threshold( img_res, 100, 255, cv2.THRESH_BINARY )
            
            # extract every individual contour area and process with dilation 
            # and threshold to make it easier to be recognized
            processed_part = single_contour_process( img_res, img_orig )
            # single_contour_process() will return 
            # => a thresholded image based on a possible contour for further recognition.
            # => None, if the ratio of height and width of result is exceeded the given limit.
            
            if processed_part is None:
                # Null return, no need to do further manipulation.
                pass
            # compare every template in mnist list
            else:
                if background_detection( processed_part ) == 1:
                    # if the background is white, turn it to black.
                    ret, processed_part = cv2.threshold( processed_part, 127, 255, cv2.THRESH_BINARY_INV )
                img_display = cv2.cvtColor( processed_part, cv2.COLOR_GRAY2BGR )
                # cv2.imshow( winname, img_display )
                # cv2.waitKey( 0 )
                
                cv2.imwrite( top_dir + SAVE_DIR + FILENAME_PREFIX + str( pic_index ) + str( count ) + FILENAME_POSFIX, img_display )
                
                feature_vector = template_matching( processed_part )
                feature_vector = np.array( feature_vector )
                
                pass_num = 0
                best_choice = -1
                for index, temp in enumerate( mnist ):
                    temp, template_vector = temp
                    template_vector = np.array( template_vector )
                    res_vector = template_vector - feature_vector
                    zero_pos = np.where( res_vector == 0 )[ 0 ]

                    if len( zero_pos ) > pass_num:
                        pass_num = len( zero_pos )
                        best_choice = index

                if best_choice > -1:
                    # print('match:', best_choice )
                    pass
                else:
                    # print('no match')
                    pass
                
                MATCH_PAIR.append( ( FILENAME_PREFIX + str( pic_index ) + str( count ) + FILENAME_POSFIX, best_choice ) )
            # the end of contour size qualified case
        # the end of comparison loop
        
    return 

def image_preprocess( img, index ):
    """
    perform basic edge enhancement and send contours to
    sample_comparison() for sample recognition. 
    """
    # winname = 'image_preprocess'
    # cv2.namedWindow( winname, cv2.WINDOW_NORMAL )

    # turn image from BGR to Gray
    gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

    # Canny process
    # edge enhancement
    blurred = cv2.GaussianBlur( gray, ( 5, 5 ), 0 )
    canny = cv2.Canny( blurred, 50, 150 )

    # dilation
    # concatenate enhanced partial edges
    if ( canny.shape[ 0 ] < 100 or canny.shape[ 1 ] < 100 ):
        kernel_size = 2
    else:
        kernel_size = 3

    dilate_kernel = np.ones( ( kernel_size, kernel_size ) )
    dilation = cv2.dilate( canny, dilate_kernel, iterations = 1 )

    # contours processing
    # find major contours on each picture
    contours, hierarchy = cv2.findContours( dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    sample_comparison( contours, hierarchy, dilation, gray, index )
    return img

def main_proc( index = 1 ):
    global top_dir
    global MATCH_PAIR

    MATCH_PAIR.clear() # reset match list
    
    top_dir = os.getcwd()
    
    set_mnist()
    
    img = read_door_img( count = index )
    if img is None:
        print( 'It is None, no such file exist.' )
        return None
    img = image_preprocess( img, index )
    cv2.destroyAllWindows()
    return img, MATCH_PAIR


if ( __name__ == '__main__' ):
    MNIST_DIR = '/static/tutor1/mnist'
    DOOR_DIR = '/static/tutor1/door'
    SAVE_DIR = '/static/tutor1/template_matching'

    index = 1

    while main_proc( index = index )[ 0 ] is not None:
        index +=1
        break
    
    # main_proc()
