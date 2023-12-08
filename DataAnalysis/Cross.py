# -*- coding: utf-8 -*-
from collections import Counter

# Form implementation generated from reading ui file 'Cross.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTableWidgetItem, QWidget


class Cross(QWidget):
    def __init__(self):
        super(Cross, self).__init__()

    def setupUi(self, row_name, col_name, row_values, col_values):
        self.setObjectName("Form")
        self.resize(600, 300)

        # 计算交叉表
        row_counter = Counter(row_values)
        col_counter = Counter(col_values)
        # 创建一个空的交叉表字典
        cross_table = {}
        # 遍历 row_group 和 col_group 中的每个元素
        for row_value, row_count in row_counter.items():
            for col_value, col_count in col_counter.items():
                # 统计同时满足条件的数量
                count = sum(
                    1 for i in range(len(row_values)) if row_values[i] == row_value and col_values[i] == col_value)
                # 将统计结果添加到交叉表中
                cross_table[(row_value, col_value)] = count

        # 获取交叉表的行索引和列索引的唯一值
        unique_rows = set(row_idx for row_idx, _ in cross_table.keys())
        unique_cols = set(col_idx for _, col_idx in cross_table.keys())
        # 创建 QTableWidget 并设置行数和列数
        row_count = len(unique_rows) + 3
        col_count = len(unique_cols) + 3
        self.tableWidget = QtWidgets.QTableWidget(self)
        self.tableWidget.setGeometry(QtCore.QRect(10, 10, 1000, 800))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(col_count)
        self.tableWidget.setRowCount(row_count)
        # self.tableWidget.setHorizontalHeaderLabels(["Chi2 Value", "P-Value"])
        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

        # 合并第1行的所有列
        for col_idx in range(col_count):
            self.tableWidget.setSpan(0, col_idx, 1, col_count)
            item = QTableWidgetItem(col_name)
            self.tableWidget.setItem(0, col_idx, item)
            item.setTextAlignment(Qt.AlignCenter)

        # 合并第0列的第2,3行
        self.tableWidget.setSpan(2, 0, 2, 1)  # setSpan(row, column, rowspan, columnspan)
        # 设置合并后单元格的值为特定字符串
        item = QTableWidgetItem(row_name)
        self.tableWidget.setItem(2, 0, item)
        item.setTextAlignment(Qt.AlignCenter)

        item = QTableWidgetItem("总计")
        self.tableWidget.setItem(row_count - 1, 0, QTableWidgetItem(item))
        item.setTextAlignment(Qt.AlignCenter)

        # 设置行标签
        for row_idx, row_value in enumerate(sorted(unique_rows), start=2):
            item = QTableWidgetItem(str(row_value))
            self.tableWidget.setItem(int(row_idx), 1, item)
            # print("行标签", int(row_idx), 1, item)

        # 设置列标签
        for col_idx, col_value in enumerate(sorted(unique_cols), start=2):
            item = QTableWidgetItem(str(col_value))
            self.tableWidget.setItem(1, int(col_idx), item)
            # print("列标签", 1, int(col_idx) + 2, item)
        self.tableWidget.setItem(1, col_count - 1, QTableWidgetItem("总计"))

        # 填充交叉表数据
        for (row_idx, col_idx), value in cross_table.items():
            item = QTableWidgetItem(str(value))
            self.tableWidget.setItem(int(row_idx) + 2, int(col_idx) + 1, item)

        # 计算每行的和并设置单元格的值
        for row_idx, row_value in enumerate(sorted(unique_rows), start=2):
            # 计算该行的和
            row_sum = sum(cross_table.get((row_value, col_value), 0) for col_value in unique_cols)
            # 将和设置为对应单元格的值
            item = QTableWidgetItem(str(row_sum))
            self.tableWidget.setItem(int(row_idx), col_count - 1, item)

        # 计算每列的和并设置单元格的值
        sums = 0
        for col_idx, col_value in enumerate(sorted(unique_cols), start=2):
            # 计算该行的和
            col_sum = sum(cross_table.get((row_value, col_value), 0) for row_value in unique_rows)
            # 将和设置为对应单元格的值
            item = QTableWidgetItem(str(col_sum))
            sums += col_sum
            self.tableWidget.setItem(row_count - 1, int(col_idx), item)

        self.tableWidget.setItem(row_count - 1, col_count - 1, QTableWidgetItem(str(sums)))

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "交叉表"))
