from plotting_window import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


WIDTH = 9
HEIGHT = 5
DPI = 100


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, pattern, parent=None, width=5, height=4, dpi=100, save=False):
        """Constructor of the class. It creates a Matplotlib canvas to plot the pattern.
        Args:
            pattern (Pattern): Pattern object to plot.
            parent (QWidget): Parent widget.
            width (int): Width of the canvas.
            height (int): Height of the canvas.
            dpi (int): Dots per inch.
            save (bool): If True, the plot will be saved in the 'saved_patterns' folder.
        """
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        self.axes.plot(pattern.dataframe_segment.iloc[:, 0])
        df2 = pattern.points
        if pattern.points is not None:
          if (isinstance(pattern.points, list)):
            for x in pattern.points:
              self.axes.plot(x)
          else:  
            self.axes.plot(df2)
        fig.suptitle(f'{pattern.company_name} {pattern.pattern_type} {pattern.starting_date[:10]} - {pattern.ending_date[:10]} - {pattern.source}')
        if pattern.tendency is True:
          self.axes.text(0.988, 0.95, '✔️', fontsize=30, color='green', ha='center', va='center', transform=self.axes.transAxes)
        elif pattern.tendency is False:
          self.axes.text(0.988, 0.95, 'X', fontsize=20, color='red', ha='center', va='center', transform=self.axes.transAxes)
        if save:
          fig.savefig(f'./saved_patterns/{pattern.company_name}_{pattern.pattern_type}_{pattern.starting_date[:10]}_{pattern.ending_date[:10]}_{pattern.source}.png')
        super(MplCanvas, self).__init__(fig)



class Grapher(QtWidgets.QWidget, Ui_Form):
  def __init__(self, *args, **kwargs):
    QtWidgets.QWidget.__init__(self, *args, **kwargs)
    self.setupUi(self)

  def plot_patterns(self, patterns_to_plot, save):
    """Method to plot the patterns in the GUI.
    Args:
        patterns_to_plot (list): List of patterns to plot.
        save (bool): If True, the plot will be saved in the 'saved_patterns' folder."""
    total_height = 0
    for pattern in patterns_to_plot:
      graph = MplCanvas(pattern, self, WIDTH, HEIGHT, DPI, save=save)
      graph.setMinimumSize(500, 400)
      self.verticalLayout_5.addWidget(graph)
      total_height += 400
    self.scrollAreaWidgetContents_4.setMinimumSize(QtCore.QSize(500, total_height))

  def add_tendency_info(self, tendency):
    """Method to add the tendency information to the GUI.
    Args:
        tendency (dict): Dictionary with the tendency information.
    """
    temp_text = ''
    for key, value in tendency.items():
      if isinstance(value[0], float):
        valueT = str(round(value[0])) + '%'
      else:
        valueT = 'Not Found'
      aux = key.replace('_', ' ')
      temp_text += f'{aux.capitalize()}: {valueT} in {str(value[1])} patterns\n'
    label = QtWidgets.QLabel()
    label.setText(temp_text)
    font = QtGui.QFont()
    font.setPointSize(12) 
    label.setFont(font)
    label.setAlignment(QtCore.Qt.AlignCenter)
    self.verticalLayout_5.addWidget(label)


if __name__ == "__main__":
  app = QtWidgets.QApplication([])
  window = Grapher()
  window.setWindowTitle("Patterns found")
  window.show()
  app.exec_()