from main_window import *
from grapher import Grapher
import re
import os
import pattern_utils
import datetime
import main as mn
from network_utils import CNN
from torch import load as torch_load
from pattern_utils import deleteRepeatedPatterns
from error_dialog import Ui_Dialog
import time


MINIMUM_DAYS = 60

class ErrorDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

def parseTxt(text):
    """
    Parses a given text file to extract company names.
    
    Args:
        text (str): The content of the text file
    
    Returns:
        list[str]: A list of company names
    """
    text = text.split()
    result_companies = []
    for word in text:
      if re.search(",$", word):
        result_companies.append(word[:-1])
      else:
        result_companies.append(word)
    return result_companies 

def compareDates(initial_date, end_date):
  """
  Compares two dates and returns the difference in days.
    
  Args:
      initial_date (QDate): The initial date
      end_date (QDate): The end date
    
  Returns:
      int: The number of days between the initial and end dates
  """
  start_date = initial_date.toPyDate()
  end_date = end_date.toPyDate()

  delta_days = (end_date - start_date).days

  return delta_days


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
  def __init__(self, *args, **kwargs):
    QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
    self.setupUi(self)
    # ventanas 
    self.current_grapher = None
    self.historic_grapher = None
    # inputs del programa
    self.companies = None
    self.isRunning = False
    self.checkboxes = [self.head_and_shoulders, self.ascending_triangle, self.descending_triangle, 
                                                            self.inv_head_and_shoulders, self.double_bottom, self.double_top]
    # botones
    self.historicpatternsbtn_both.clicked.connect(self.PreRunProgram)
    self.currentpatternsbtn_both.clicked.connect(self.PreRunProgram)
    self.openfilebtn.clicked.connect(self.openTxt)
    # barra de progreso
    self.progressBar.reset()
    self.progressBar.setMinimum(0)
    self.progressBar.setHidden(True)
    self.model_double_top = CNN()
    self.model_double_bottom = CNN()
    self.model_ascending_triangle = CNN()
    self.model_descending_triangle = CNN()
    self.model_head_and_shoulders = CNN()
    self.model_inv_head_and_shoulders = CNN()
    self.model_double_top.load_state_dict(torch_load('double_top_model.pth'))
    self.model_double_bottom.load_state_dict(torch_load('double_bottom_model.pth'))
    self.model_ascending_triangle.load_state_dict(torch_load('ascending_triangle_model.pth'))
    self.model_descending_triangle.load_state_dict(torch_load('descending_triangle_model.pth'))
    self.model_head_and_shoulders.load_state_dict(torch_load('head_and_shoulders_model.pth'))
    self.model_inv_head_and_shoulders.load_state_dict(torch_load('inv_head_and_shoulders_model.pth'))


  def PreRunProgram(self):
    """
    Prepares and runs the program based on user inputs and selected options.
    """
    if self.isRunning:
      print('Already running')
      return
    if self.companies == None:
      print('Select companies')
      return
    if not isinstance(self.initialdate.date().toPyDate(), datetime.date) and not isinstance(self.enddate.date().toPyDate(), datetime.date):
      raise Exception('Enter a valid year format %dddd')
    selected_types_set = set()
    self.models = {}
    for checkbox in self.checkboxes:
      if checkbox.isChecked():
        selected_types_set.add(checkbox.objectName())
        self.models[checkbox.objectName()] = getattr(self, f'model_{checkbox.objectName()}')
    if compareDates(self.initialdate.date(), self.enddate.date()) < MINIMUM_DAYS and (self.sender() == self.historicpatternsbtn_both):
      dialog = ErrorDialog()
      dialog.exec_()
    else:
      self.runProgram(self.sender(), selected_types_set)

  def runProgram(self, button, selected_types):
    """
    Runs the program to analyze patterns based on selected options.
        
    Args:
        button (QPushButton): The button that was clicked to start the program
        selected_types (set[str]): The set of selected pattern types
    """
    self.isRunning = True
    self.progressBar.setHidden(False)
    if button is None or not isinstance(button, QtWidgets.QPushButton):
        return  # Safety check
    patterns_dictionary = pattern_utils.loadPatterns(15, selected_types)
    historic_results = []
    current_results = []
    window_sizes = self.get_window_sizes(self.comboBox.currentText())
    search_mode = self.h_search_type.currentText()
    self.progressBar.setMaximum(len(self.companies))
    company_index = 0
    for company in self.companies:
      print(company)
      company_index += 1
      if button == self.historicpatternsbtn_both:
        historic_results = historic_results + self.trainHistoric(company, patterns_dictionary, self.initialdate.date().toPyDate(), self.enddate.date().toPyDate(), window_sizes, search_mode)
      elif button == self.currentpatternsbtn_both:
        current_results = current_results + self.findCurrent(company, patterns_dictionary, window_sizes, search_mode)
      self.progressBar.setValue(company_index)
    self.isRunning = False
    self.progressBar.reset()
    if button == self.currentpatternsbtn_both:
      self.show_current_patterns(current_results)
    else:
      tendency_results = pattern_utils.calculateTendencyProbability(historic_results, selected_types)
      self.show_historic_patterns(historic_results, tendency_results)
    self.progressBar.setHidden(True)

  def trainHistoric(self, company, patterns_dictionary, initial_date, end_date, window_sizes, search_mode):
    """
    Search for historic patterns
    Args:
        company (str): The company name
        patterns_dictionary (dict): The dictionary of patterns
        initial_date (datetime.date): The initial date
        end_date (datetime.date): The end date
        window_sizes (list[int]): The list of window sizes
        search_mode (str): The search mode
    Returns:
        list[Pattern]: The list of historic pattern results"""
    historic_results = None
    if search_mode == "DTW and Neural Network":
      historic_results = mn.trainHistoricDatabaseNetwork(company, patterns_dictionary, initial_date, end_date, self.models, window_sizes)
      historic_results = historic_results + mn.trainHistoricDatabaseAutomatic(company, patterns_dictionary, initial_date, end_date, window_sizes)
      historic_results = deleteRepeatedPatterns(historic_results)
    elif search_mode == "DTW":
      historic_results = mn.trainHistoricDatabaseAutomatic(company, patterns_dictionary, initial_date, end_date, window_sizes)
    elif search_mode == "Neural Network":
      historic_results = mn.trainHistoricDatabaseNetwork(company, patterns_dictionary, initial_date, end_date, self.models, window_sizes)
    return historic_results
  
  def findCurrent(self, company, patterns_dictionary, window_sizes, search_mode):
    current_results = None
    if search_mode == "DTW and Neural Network":
      current_results = mn.findCurrentPatternsAutomatic(company, patterns_dictionary, window_sizes)
      current_results = current_results + mn.findCurrentPatternsNetwork(company, self.models, window_sizes)
      current_results = deleteRepeatedPatterns(current_results)
    elif search_mode == "DTW":
      current_results = mn.findCurrentPatternsAutomatic(company, patterns_dictionary, window_sizes)
    elif search_mode == "Neural Network":
      current_results = mn.findCurrentPatternsNetwork(company, self.models, window_sizes)
    return current_results

  def show_historic_patterns(self, historic_results, tendency_results):
    """
      Displays the historic patterns and their tendency results.
        
      Args:
          historic_results (list): The list of historic pattern results
          tendency_results (list): The list of tendency results
    """
    self.historic_grapher = Grapher()
    self.historic_grapher.add_tendency_info(tendency_results)
    self.historic_grapher.plot_patterns(historic_results, self.save_patterns.isChecked())
    self.historic_grapher.show()
  
  def show_current_patterns(self, current_results):
    """
    Displays the current patterns.
        
    Args:
        current_results (List[Pattern]): The list of current pattern results
    """
    self.current_grapher = Grapher()
    self.current_grapher.plot_patterns(current_results, self.save_patterns.isChecked())
    self.current_grapher.show()
  
  def openTxt(self):
    """
    Opens a text file and parses the content to extract company names.
    """
    options = QtWidgets.QFileDialog.Options()
    fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Text File", "", "Text Files (*.txt)", options=options)
    if fileName:
      with open(fileName, 'r') as file:
        input_text = file.read()
        self.selectedfile.setText('Selected File: ' + os.path.basename(fileName))
        self.companies = parseTxt(input_text)

  def get_window_sizes(self, text):
    """
    Returns the list of window_sizes based on the selected search type.
        
    Args:
      text (str): The selected search type
        
    Returns:
      list[int]: The list of window_sizes for the search type
    """
    if text == 'Normal Search':
      return [80, 120, 160, 200, 240]
    elif text == 'Fast Search':
      return [80, 120, 200]
    elif text == 'Deep Search':
      return  [80, 100, 120, 140, 160, 180, 200, 220, 240]
    elif text == 'Very Deep Search':
      return  [80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260]
    elif text == 'Small Patterns Search':
      return  [20, 30, 40, 50, 60, 70]
    elif text == 'Big Patterns Search':
      return [220, 240, 260, 280, 300]
    else:
      raise Exception('Invalid search type')

if __name__ == "__main__":
  app = QtWidgets.QApplication([])
  window = MainWindow()
  window.setWindowTitle('PerseumAI')
  window.show()
  app.exec_()
