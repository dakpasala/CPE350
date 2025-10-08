import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import random
from parse_output2 import parse_xml

for frame in parse_xml("output1.xml"):
    print(frame)


frames = list(parse_xml("output1.xml"))
frame_index = 0

# Initialize Qt app
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Bosch Traffic Map (Live)")
plot = win.addPlot()
plot.setLabel('left', 'Latitude offset')
plot.setLabel('bottom', 'Longitude offset')
plot.setAspectLocked(True)  # Equal scale
plot.showGrid(x=True, y=True)
plot.setRange(xRange=(-1, 1), yRange=(-1, 1))

# Define scatter groups
car_scatter = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 0, 0, 180))
truck_scatter = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(0, 0, 255, 180))
person_scatter = pg.ScatterPlotItem(size=8, brush=pg.mkBrush(0, 255, 0, 180))
plot.addItem(car_scatter)
plot.addItem(truck_scatter)
plot.addItem(person_scatter)

# Scaling for better visualization
SCALE = 1e5

def update():
    global frame_index
    if frame_index >= len(frames):
        frame_index = 0  # loop for demo
    frame = frames[frame_index]
    frame_index += 1
    
    cars = [(obj["lon"] * SCALE, obj["lat"] * SCALE) for obj in frame if obj["type"] == "Car"]
    trucks = [(obj["lon"] * SCALE, obj["lat"] * SCALE) for obj in frame if obj["type"] == "Truck"]
    persons = [(obj["lon"] * SCALE, obj["lat"] * SCALE) for obj in frame if obj["type"] == "Person"]

    car_scatter.setData([x for x, y in cars], [y for x, y in cars])
    truck_scatter.setData([x for x, y in trucks], [y for x, y in trucks])
    person_scatter.setData([x for x, y in persons], [y for x, y in persons])

# Timer for ~24 FPS
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1000 // 24)

QtWidgets.QApplication.instance().exec()
