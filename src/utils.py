import numpy
class RLECoder():
    def __init__(self):
        pass
    
    def encode(self, binary_image):
        start = None
        length = None
        
        result = [*binary_image.shape]

        for index in range(len(binary_image.flat)):

            if binary_image.flat[index] > 0.5:
                if start is None:
                    start = index
                    length = 1

                else:
                    length += 1
            else:
                if start is not None:
                    result.append(start + 1)
                    result.append(length)
                    start = None
                    length = None
        if start is not None:
            result.append(start + 1)
            result.append(length)
        return result

    def decode(self, rle):
        shape = rle[:2]
        code = rle[2:]
        image = numpy.zeros(shape)
        image = image.flatten()

        for index in range(0, len(code), 2):
            start = code[index] - 1
            length = code[index + 1]
            image[start:start+length] = 1

        image = image.reshape(shape)
        return image
