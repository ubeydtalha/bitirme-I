import os
from PyLTSpice.LTSpiceBatch import *
from PyLTSpice.LTSteps import LTSpiceLogReader

meAbsPath = os.path.dirname(os.path.realpath(__file__))
LTC = SimCommander(meAbsPath + "\\Draft1.asc")

LTC.set_parameter("R","5k")

LTC.add_instruction(".tran 0 2m")

def processing_data(raw_filename, log_filename):
    '''This is a call back function that just prints the filenames'''
    print("Simulation Raw file is %s. The log is %s" % (raw_filename, log_filename))
    # Other code below either using LTSteps.py or LTSpice_RawRead.py
    log_info = LTSpiceLogReader(log_filename)
    log_info.read_measures()
    rise, measures = log_info.dataset["rise_time"]

LTC.run(callback=processing_data)
LTC.wait_completion()

