# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'K2_result.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTableWidgetItem

from DataAnalysis.test import chi_square_test


class K2_result(QtWidgets.QWidget):
    def __init__(self):
        super(K2_result, self).__init__()

    def setupUi(self, row_values, col_values):
        self.setObjectName("Form")
        self.resize(1000, 300)
        self.tableWidget = QtWidgets.QTableWidget(self)
        self.tableWidget.setGeometry(QtCore.QRect(10, 10, 1000, 800))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setRowCount(6)
        # self.tableWidget.setHorizontalHeaderLabels(["Chi2 Value", "P-Value"])
        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

        results = chi_square_test(row_values, col_values)

        self.tableWidget.setItem(1, 1, QTableWidgetItem('值'))
        self.tableWidget.setItem(1, 2, QTableWidgetItem('自由度'))
        self.tableWidget.setItem(1, 3, QTableWidgetItem('渐进显著性(双侧)'))
        # self.tableWidget.setItem(1, 4, QTableWidgetItem('精确显著性(双侧)'))
        # self.tableWidget.setItem(1, 5, QTableWidgetItem('精确显著性(单侧)'))

        self.tableWidget.setItem(2, 0, QTableWidgetItem('皮尔逊卡方'))
        # self.tableWidget.setItem(3, 0, QTableWidgetItem('连续性修正'))
        self.tableWidget.setItem(3, 0, QTableWidgetItem('似然比'))
        # self.tableWidget.setItem(5, 0, QTableWidgetItem('费希尔精确检验'))
        self.tableWidget.setItem(4, 0, QTableWidgetItem('线性关联'))
        self.tableWidget.setItem(5, 0, QTableWidgetItem('有效个案数'))

        # 皮尔逊卡方
        Pearson_Chi_Square = results['Pearson Chi-square']
        for i, (attr, attr_value) in enumerate(Pearson_Chi_Square.items()):
            item = QTableWidgetItem(str(attr_value))
            self.tableWidget.setItem(2, i + 1, item)

        # 连续性修正
        # Continuity_Correction = results['Continuity Correction']
        # for i, (attr, attr_value) in enumerate(Continuity_Correction.items()):
        #     item = QTableWidgetItem(str(attr_value))
        #     self.tableWidget.setItem(3, i + 1, item)

        # 似然比
        Likelihood_Ratio = results['Likelihood Ratio']
        for i, (attr, attr_value) in enumerate(Likelihood_Ratio.items()):
            item = QTableWidgetItem(str(attr_value))
            self.tableWidget.setItem(3, i + 1, item)

        # 费希尔精确检验
        # Fisher_Exact_Test = results['Fisher Exact Test']
        # for i, (attr, attr_value) in enumerate(Fisher_Exact_Test.items()):
        #     item = QTableWidgetItem(str(attr_value))
        #     self.tableWidget.setItem(5, i + 1, item)

        # 线性关联
        Linear_Correlation = results['Linear Correlation']
        for i, (attr, attr_value) in enumerate(Linear_Correlation.items()):
            item = QTableWidgetItem(str(attr_value))
            self.tableWidget.setItem(4, i + 1, item)

        # 有效个案数
        Valid_Cases = results['Valid Cases']
        for i, (attr, attr_value) in enumerate(Valid_Cases.items()):
            item = QTableWidgetItem(str(attr_value))
            self.tableWidget.setItem(5, i + 1, item)

        # Values = results['Value']
        # for i, (attr, attr_value) in enumerate(Values.items()):
        #     item = QTableWidgetItem(str(attr_value))
        #     self.tableWidget.setItem(i+2, 1, item)
        #
        # Degrees_of_Freedom = results['Degrees of Freedom']
        # for i, (attr, attr_value) in enumerate(Degrees_of_Freedom.items()):
        #     item = QTableWidgetItem(str(attr_value))
        #     self.tableWidget.setItem(i+2, 2, item)
        # Asymptotic_p_value_two_side = results['Asymptotic_p_value_two_side']
        # for i, (attr, attr_value) in enumerate(Asymptotic_p_value_two_side.items()):
        #     item = QTableWidgetItem(str(attr_value))
        #     self.tableWidget.setItem(i+2, 3, item)
        # Exact_p_value_two_sided=results['Exact_p_value_two_sided']
        # for i, (attr, attr_value) in enumerate(Exact_p_value_two_sided.items()):
        #     item = QTableWidgetItem(str(attr_value))
        #     self.tableWidget.setItem(i+2, 4, item)
        # Exact_p_value_one_sided = results['Exact_p_value_one_sided']
        # for i, (attr, attr_value) in enumerate(Exact_p_value_one_sided.items()):
        #     item = QTableWidgetItem(str(attr_value))
        #     self.tableWidget.setItem(i+2, 5, item)

        # 合并第0行的所有列
        self.tableWidget.setSpan(0, 0, 1, 6)  # setSpan(row, column, rowspan, columnspan)
        item = QTableWidgetItem('卡方检验')
        self.tableWidget.setItem(0, 0, item)
        item.setTextAlignment(Qt.AlignCenter)

        #所有单元格居中对齐
        for row in range(self.tableWidget.rowCount()):
            for column in range(self.tableWidget.columnCount()):
                self.tableWidget.setColumnWidth(column, 150)  # 第column列宽度为100像素
                item = self.tableWidget.item(row, column)
                if item is not None:
                    item.setTextAlignment(Qt.AlignCenter)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "卡方检验"))
