import matplotlib.pyplot as plt
import math

class FractalTree:

    def __init__(self):
        plt.title("Fractal Tree")
        plt.figure(1)
        plt.axis((0, 600, 600, 0))

    def drawTree(self, x1, y1, angle, depth):
        if depth:
            x2 = x1 + int(math.cos(math.radians(angle)) * depth * 10.0)
            y2 = y1 + int(math.sin(math.radians(angle)) * depth * 10.0)
            plt.plot([x1, x2],  [y1, y2], 'g-')
            print(x1, y1, x2, y2)
            self.drawTree(x2, y2, angle - 20, depth - 1)
            self.drawTree(x2, y2, angle + 20, depth - 1)


if __name__ == "__main__":
    ft = FractalTree()
    ft.drawTree(300, 550, -90, 9)
    plt.show()
