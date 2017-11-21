import matplotlib.pyplot as plot
import numpy
import struct

def main():
    image_file = open('mnist-train-images-ubyte', 'rb')
    magic, count, rows, columns = struct.unpack('>IIII', image_file.read(16))
    pic0 = image_file.read(28*28)
    pic0 = struct.unpack('784B', pic0)
    img = numpy.array(pic0)
    img = img.reshape(28, 28)
    plot.figure()
    plot.imshow(img)
    plot.show()
    image_file.close()
    

if __name__ == '__main__':
    main()

