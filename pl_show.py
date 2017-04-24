import pylab as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
x=[x for x in range(100)]
y = [x**2 for x in range(200) if x%2 == 0 ]
z = [x*3 for x in range(100)]
figure = plt.figure()
canvas = FigureCanvas(figure)
figure.add_axes()
ax = Axes3D()
figure.add_axes(ax)
ax.plot(x,y,z,'o')
canvas.print_figure('demo.jpg')
plt.show()