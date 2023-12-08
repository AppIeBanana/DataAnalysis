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

from DataAnalysis.test import f_test_anova


class F_anova(QtWidgets.QWidget):
    def __init__(self):
        super(F_anova, self).__init__()

    def setupUi(self, row_name, row_values, col_values):
        self.setObjectName("单因素F(ANOVA)检验结果")
        self.resize(642, 344)
        self.tableWidget = QtWidgets.QTableWidget(self)
        self.tableWidget.setGeometry(QtCore.QRect(0, 0, 641, 341))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(6)
        self.tableWidget.setRowCount(6)

        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

        results = f_test_anova(row_values, col_values)
        #列宽填满
        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        # 合并第0行的所有列
        self.tableWidget.setSpan(0, 0, 1, 6)  # setSpan(row, column, rowspan, columnspan)
        item = QTableWidgetItem('ANOVA')
        self.tableWidget.setItem(0, 0, item)
        item.setTextAlignment(Qt.AlignCenter)

        # 合并第1行的1-5列
        self.tableWidget.setSpan(1, 1, 1, 5)  # setSpan(row, column, rowspan, columnspan)
        self.tableWidget.setItem(1, 0, QTableWidgetItem(''))

        self.tableWidget.setItem(2, 1, QTableWidgetItem('平方和'))
        self.tableWidget.setItem(2, 2, QTableWidgetItem('自由度'))
        self.tableWidget.setItem(2, 3, QTableWidgetItem('均方'))
        self.tableWidget.setItem(2, 4, QTableWidgetItem('F'))
        self.tableWidget.setItem(2, 5, QTableWidgetItem('显著性'))

        self.tableWidget.setItem(1, 0, QTableWidgetItem(str(row_name)))
        self.tableWidget.setItem(3, 0, QTableWidgetItem('组间'))
        self.tableWidget.setItem(4, 0, QTableWidgetItem('组内'))
        self.tableWidget.setItem(5, 0, QTableWidgetItem('总计'))

        for row in range(0, 3):
            for col in range(0, 5):
                self.tableWidget.setItem(row + 3, col + 1, QTableWidgetItem(str(results[row][col])))

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))