import sys
import os
import pickle

import numpy as np
import xlrd

def readxlsx(file):
    try:
        book = xlrd.open_workbook(os.path.join(os.getcwd(),"excels",file))
    except FileNotFoundError:
        print("sorry, readxlsx could't find a file")
        print("End program")
        sys.exit()
    sheet = {}
    col_start = 1
    col_end = 50
    for i in range(1,len(book.sheets())+1):
        sheet[i] = book.sheet_by_name("Sheet{}".format(i))
        x_data = [sheet[1].row_values(x,0,3) for x in range(col_start,col_end)]
        x_data = np.asarray(x_data)
        t_data = [[1.0,0.0,0.0] if sheet[1].row_values(x,4,5)[0]==0.0 else [0.0,1.0,0.0] if sheet[1].row_values(x,4,5)[0]==1.0 else [0.0,0.0,1.0] for x in range(col_start,col_end)]
        #answer_data = [sheet[1].row_values(x,4,5) for x in range(row_start,row_end)]
        t_data = np.asarray(t_data)
        print("loading xlsx has completed!\n")
    return x_data,t_data
