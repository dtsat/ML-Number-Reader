import numpy
import cv2
from base import DescriptorBase
class ThreeByThreeDescriptor(DescriptorBase):
    """
    `A New Clustering Method for Improving Plasticity and Stability in Handwritten
    Character Recognition Systems`
    """

    def __init__(self, data=None):
        """
        Constructor.

        Parameters:
            data (optional) -- numpy.array of 225
        """
        if data is None:
            super(ThreeByThreeDescriptor, self).__init__(size=225)
        else:
            assert len(data) == 225
            super(ThreeByThreeDescriptor, self).__init__(data)
    

# Finished, use me
class ThreeByThreeFeatureExtractor:
    """
    `A New Clustering Method for Improving Plasticity and Stability in Handwritten
    Character Recognition Systems`
    """
    def __init__(self):
        """
        Constructor
        """
        pass

    def __del__(self):
        """
        Destructor
        """
        pass

    def __str__(self):
        """
        toString(), like in Java
        """
        return "Three By Three Feature Extractor"

    
    def __white(self, row, col):
        """
        Set 3x3 to white
        """
        for x in range(row, row+2, 1):
            for y in range(col, col+2, 1):
                self.smooth[x][y] = 1.0
    
    def __black(self, row, col):
        """
        Set 3x3 to white except the center which is set to black
        """
        self.__white(row, col)
        self.smooth[row+1][col+1] = 0.0

    def __check(self, row, col, white_threshold):
        for x in range(row, row+2, 1):
            for y in range(col, col+2, 1):
                val = self.work[x][y]
                # Value is too bright, assuming white
                if val > white_threshold:
                    continue
                # Black enough for us
                else:
                    return True
        return False


    def __make_it_smooth(self, white_threshold=0.9):
        """
        As per the paper, smooth the image by removing extra black in 3x3 steps.
        
        Keyword Arguments:
        white_threshold -- Any color below this threshold is considered black, above is white.
        We have good results with 0.9 (out of 1)
        """
        if not self.image.data:
            print("FeatureExtractor::make_it_smooth Load image first. Usage: extractor = FeatureExtractor(image)")
            exit()
        if self.image.shape is not (45, 45):
            # Resize image to 45x45 as per paper 'ICPR clustering'
            self.work = cv2.resize(self.image, (45, 45))
            self.smooth = numpy.ones((45, 45))
        else:
            self.work = self.image
            self.smooth = numpy.ones((45, 45))

        # Smooth
        for row in range(0, 45, 3):
            rowC = 0
            for col in range(0, 45, 3):
                hasBlack = self.__check(row, col, white_threshold)
                if hasBlack:
                    self.__black(row, col)
                else:
                    self.__white(row, col)
        self.feature = cv2.resize(self.smooth, (15, 15))
        

    def run(self, image):
        """
        Parameters:
            image -- OpenCV image
        """
        self.image = image
        self.__make_it_smooth()
        self.result = []
        for y in range(15):
            for x in range(15):
                self.result.append(self.feature[x][y])
        return ThreeByThreeDescriptor(numpy.array(self.result))

