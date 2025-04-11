

from vp import *
from dp import Ui_Dialog

from Pattern import Pattern

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QScrollArea
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.dates as mdates

class MyForm(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        # Configura el layout principal y el área de desplazamiento
        self.main_layout = QVBoxLayout(self)
        self.scroll_area = QScrollArea(self)
        self.scroll_area_widget_contents = QWidget()
        self.scroll_area_layout = QVBoxLayout(self.scroll_area_widget_contents)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.scroll_area_widget_contents)
        self.main_layout.addWidget(self.scroll_area)

    def showPatterns(self, results):
        """Show patterns given as a list"""
        for pattern in results:
            # Crea un widget para cada gráfico y configura su layout
            chart_widget = QWidget(self.scroll_area_widget_contents)
            chart_layout = QVBoxLayout(chart_widget)

            fig = Figure(figsize=(9, 5), dpi=100)
            plot1 = fig.add_subplot(111)
            plot1.plot(pattern.dataframe_segment.iloc[:, 0])

            df2 = pattern.points
            if pattern.points is not None:
                if isinstance(pattern.points, list):
                    for x in pattern.points:
                        plot1.plot(x)
                else:
                    plot1.plot(df2)

            fig.suptitle(f'{pattern.company_name} {pattern.pattern_type} {pattern.starting_date[:10]} - {pattern.ending_date[:10]}')
            canvas = FigureCanvas(fig)
            chart_layout.addWidget(canvas)

            if pattern.tendency is True:
                tendency = '✔️'
                pattern_tendency_text = QTextEdit(tendency, chart_widget)
                pattern_tendency_text.setStyleSheet("background-color: #40BD2E")
            elif pattern.tendency is False:
                tendency = '❌'
                pattern_tendency_text = QTextEdit(tendency, chart_widget)
                pattern_tendency_text.setStyleSheet("background-color: red")
            else:
                continue

            pattern_tendency_text.setReadOnly(True)
            chart_layout.addWidget(pattern_tendency_text)

            # Ajustar las fechas del eje X
            for ax in fig.axes:
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
                ax.tick_params(axis='x', rotation=30)

            fig.tight_layout()

            # Agregar el widget del gráfico al layout del área de desplazamiento
            self.scroll_area_layout.addWidget(chart_widget)


    def showPatterns2(self, data_lists):
        """Show patterns given as a list of lists"""
        for data in data_lists:
            # Crea un widget para cada gráfico y configura su layout
            chart_widget = QWidget(self.scroll_area_widget_contents)
            chart_layout = QVBoxLayout(chart_widget)

            fig = Figure(figsize=(9, 5), dpi=100)
            plot1 = fig.add_subplot(111)
            plot1.plot(data)

            canvas = FigureCanvas(fig)
            chart_layout.addWidget(canvas)

            fig.tight_layout()
            '''
            tendency_text = QTextEdit(chart_widget)
            if data[-1] >= 5:
                tendency_text.setText('✔️')
                #tendency_text.setStyleSheet("background-color: #40BD2E")
            else:
                tendency_text.setText('❌')
                tendency_text.setStyleSheet("background-color: red")

            tendency_text.setReadOnly(True)
            chart_layout.addWidget(tendency_text)
            '''
            # Agregar el widget del gráfico al layout del área de desplazamiento
            self.scroll_area_layout.addWidget(chart_widget)

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
  def __init__(self, *args, **kwargs):
    QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
    self.setupUi(self)
    self.pushButton_2.clicked.connect(self.show_dialog)

  def show_dialog(self):
    self.Dialog = QtWidgets.QDialog()
    self.ui = Ui_Dialog()
    self.ui.setupUi(self.Dialog)
    self.Dialog.show()
  


if __name__ == "__main__":
  app = QtWidgets.QApplication([])
  #window = MainWindow()
  #crear una lista de patterns para comprobar que funciona
  window = MyForm()
  test_data = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [2, 3, 1, 4, 5]]
  window.showPatterns2(test_data)
  window.show()
  app.exec_()