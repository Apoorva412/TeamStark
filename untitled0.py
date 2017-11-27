#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:10:40 2017

@author: manniarora
"""
import csv
#----------------------------------------------------------------------
def csv_writer(data, path):
    """
    Write data to a CSV file path
    """
    with open(path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            print(line)
            writer.writerow(line)
def csv_reader(file_obj):
    """
    Read a csv file
    """
    reader = csv.reader(file_obj,delimiter=',')
    for row in reader:
        print(" ".join(row))
if __name__ == "__main__":
    data = ["first_name,last_name,city".split(","),
            "Tyrese,Hirthe,Strackeport".split(","),
            "Jules,Dicki,Lake Nickolasville".split(","),
            "Dedric,Medhurst,Stiedemannberg".split(",")
            ]
    path = "/Users/manniarora/Desktop/resume-parser-new/output.csv"
    csv_writer(data,path)
    with open(path, "r") as f_obj:
        csv_reader(f_obj)
           
