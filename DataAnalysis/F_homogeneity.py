# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'F_homogeneity.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTableWidgetItem

from DataAnalysis.test import f_test_homogeneity


class F_homogeneity(QtWidgets.QWidget):
    def __init__(self):
        super(F_homogeneity, self).__init__()

    def setupUi(self, row_name, row_values, col_values,states):
        self.setObjectName("方差齐性检验结果")
        self.resize(630, 300)
        self.tableWidget = QtWidgets.QTableWidget(self)
        self.tableWidget.setGeometry(QtCore.QRect(0, 0, 641, 341))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(6)
        self.tableWidget.setRowCount(6)

        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

        results = f_test_homogeneity(row_values, col_values)
        #列宽填满
        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        # 合并第0行的所有列
        self.tableWidget.setSpan(0, 0, 1, 6)  # setSpan(row, column, rowspan, columnspan)
        item = QTableWidgetItem('方差齐性检验')
        self.tableWidget.setItem(0, 0, item)
        item.setTextAlignment(Qt.AlignCenter)

        # 合并第0列的2-6行
        self.tableWidget.setSpan(1, 0, 5, 1)  # setSpan(row, column, rowspan, columnspan)
        item = QTableWidgetItem(row_name)
        self.tableWidget.setItem(1, 0, QTableWidgetItem(item))
        item.setTextAlignment(Qt.AlignCenter)

        self.tableWidget.setItem(1, 2, QTableWidgetItem('莱文统计'))
        self.tableWidget.setItem(1, 3, QTableWidgetItem('自由度1'))
        self.tableWidget.setItem(1, 4, QTableWidgetItem('自由度2'))
        self.tableWidget.setItem(1, 5, QTableWidgetItem('显著性'))

        self.tableWidget.setItem(2, 1, QTableWidgetItem('基于平均值'))
        self.tableWidget.setItem(3, 1, QTableWidgetItem('基于中位数'))
        self.tableWidget.setItem(4, 1, QTableWidgetItem('基于中位数并具有调整后自由度'))
        self.tableWidget.setItem(5, 1, QTableWidgetItem('基于检出后'))

        for row in range(0, 4):
            for col in range(0, 4):
                self.tableWidget.setItem(row + 2, col + 2, QTableWidgetItem(str(results[row][col])))

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))