import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import stats
from scipy.stats import f_oneway, median_test, trim_mean, levene


# selected_items中的所有项都是excel表中对应的列，submit函数的作用是将selected_items中的所有项，也就是excel表中该列的值
# 单样本 T 检验
def single_sample_t_test(column_value, population_mean, confidence_level):
    column_value = pd.Series(column_value)  # 将列表转换为 Series 对象
    n = len(column_value)
    sample_mean = column_value.mean()
    sample_std = column_value.std(ddof=1)  # 使用样本标准差，ddof(自由度)为 n-1

    t_statistic, p_value = stats.ttest_1samp(column_value, population_mean)

    # 计算置信区间
    margin_of_error = stats.t.ppf((1 + confidence_level) / 2, df=n - 1) * (sample_std / (n ** 0.5))  # 误差边界=t分布临界值*标准误差
    confidence_low = sample_mean - margin_of_error - population_mean  # 置信区间下限
    confidence_hign = sample_mean + margin_of_error - population_mean  # 置信区间上限
    result = []
    result.append(round(t_statistic, 3))
    result.append(n - 1)
    result.append(round(p_value, 3))
    result.append(round(sample_mean - population_mean, 3))
    result.append(round(confidence_low, 2))
    result.append(round(confidence_hign, 2))
    return result


# 独立样本 T 检验
def independent_samples_t_test(data, column_name, group_column, group1, group2, confidence_level=0.95):
    group1_data = data[data[group_column] == group1][column_name]
    group2_data = data[data[group_column] == group2][column_name]
    n1 = len(group1_data)
    n2 = len(group2_data)
    # 方差等同性检验
    _, p_value_levene = stats.levene(group1_data, group2_data)

    if p_value_levene >= 0.05:  # 方差等同性检验通过，假定等方差
        t_statistic, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=True)

        df = n1 + n2 - 2
        sample_mean_diff = group1_data.mean() - group2_data.mean()
        pooled_std = ((n1 - 1) * group1_data.var(ddof=1) + (n2 - 1) * group2_data.var(ddof=1)) / df
        margin_of_error = stats.t.ppf((1 + confidence_level) / 2, df=df) * (pooled_std * ((1 / n1) + (1 / n2))) ** 0.5
        confidence_interval = (sample_mean_diff - margin_of_error, sample_mean_diff + margin_of_error)

        return t_statistic, p_value, confidence_interval, "Equal variances assumed (Levene's test p-value: {:.4f})".format(
            p_value_levene)
    else:  # 方差等同性检验不通过，假定不等方差
        t_statistic, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)

        # 计算置信区间
        sample_mean_diff = group1_data.mean() - group2_data.mean()
        std1 = group1_data.std(ddof=1)
        std2 = group2_data.std(ddof=1)
        dof = (std1 ** 2 / n1 + std2 ** 2 / n2) ** 2 / (
                (std1 ** 2 / n1) ** 2 / (n1 - 1) + (std2 ** 2 / n2) ** 2 / (n2 - 1))
        margin_of_error = stats.t.ppf((1 + confidence_level) / 2, df=int(dof)) * (
                std1 ** 2 / n1 + std2 ** 2 / n2) ** 0.5
        confidence_interval = (sample_mean_diff - margin_of_error, sample_mean_diff + margin_of_error)

        # 输出各个返回值的结果
        print("T-statistic:", t_statistic)
        print("P-value:", p_value)
        print("Confidence Interval:", confidence_interval)

        return t_statistic, p_value, confidence_interval


def paired_samples_t_test(data, column_name1, column_name2, confidence_level=0.95):
    sample1 = data[column_name1]
    sample2 = data[column_name2]

    # 计算 Pearson 相关系数
    correlation_coefficient, _ = stats.pearsonr(sample1, sample2)

    # 执行配对样本 T 检验
    diff = sample1 - sample2
    t_statistic, p_value = stats.ttest_rel(sample1, sample2)

    # 计算置信区间的范围
    n = len(diff)
    df = n - 1
    critical_value = stats.t.ppf((1 + confidence_level) / 2, df=df)
    standard_error = np.std(diff, ddof=1) / np.sqrt(n)  # 标准误差
    margin_of_error = critical_value * standard_error  # 误差边界
    sample_mean_diff = np.mean(diff)  # 样本平均值差
    confidence_interval = (sample_mean_diff - margin_of_error, sample_mean_diff + margin_of_error)  # 置信度

    # 输出各个返回值的结果
    print("Pearson correlation coefficient:", correlation_coefficient)
    print("T-statistic:", t_statistic)
    print("P-value:", p_value)
    print("Degrees of freedom:", df)
    print("Critical value:", critical_value)
    print("Standard error:", standard_error)
    print("Margin of error:", margin_of_error)
    print("Sample mean difference:", sample_mean_diff)
    print("Confidence interval:", confidence_interval)

    # 返回结果  Pearson相关系数，配对样本 T 检验的 T 统计量，配对样本 T 检验的 p 值，自由度，临界值，标准误差，误差边界，样本平均值差，置信区间
    return correlation_coefficient, t_statistic, p_value, df, critical_value, standard_error, margin_of_error, sample_mean_diff, confidence_interval


def f_test_homogeneity(row_values, col_values):
    # row_values因变量  col_values因子
    row_values = list(map(float, row_values))
    col_values = list(map(float, col_values))
    print('---------------------方差齐性检验------------------------')
    print('因变量:', row_values)
    print('因子:', col_values)
    # 计算基于平均值莱文统计量和显著性
    w_mean, p_w_mean = levene(row_values, col_values, center='mean')
    w_mean, p_w_mean = round(w_mean, 3), round(p_w_mean, 3)
    # 计算基于中位数的莱文统计量和显著性
    w_median, p_w_median = levene(row_values, col_values, center="median")
    w_median, p_w_median = round(w_median, 3), round(p_w_median, 3)
    # 计算基于剪除后平均值的莱文统计量和显著性
    w_median_adj, p_w_median_adj = levene(row_values, col_values, center="trimmed")
    w_median_adj, p_w_median_adj = round(w_median_adj, 3), round(p_w_median_adj, 3)
    # 计算自由度
    df1 = len(set(col_values)) - 1
    df2 = len(row_values) - 1
    # 创建一个新的row_values列表，只包含非0的元素
    new_row_values = row_values
    new_col_values = col_values
    new_row_values = [x for x in new_row_values if x != 0]
    # 创建一个新的col_values列表，只包含与row_values中非0元素对应的元素
    new_col_values = [new_col_values[n] for n, x in enumerate(new_row_values) if x != 0]
    # 计算基于中位数的莱文统计量和显著性
    _w_median, _p_w_median = levene(new_row_values, new_col_values, center="median")
    _w_median, _p_w_median = round(_w_median, 3), round(_p_w_median, 3)
    # 计算新的自由度
    _df1 = len(set(new_col_values)) - 1
    _df2 = len(new_row_values) - 1
    print('原自由度:', df1, df2)
    print('新的自由度:', _df1, _df2)
    print('剪除后的因变量:', new_row_values)
    print('剪除后的因子:', new_col_values)
    print('-------------------------------------------------------')
    # 创建输出表格
    output = [
        [w_mean, df1, df2, p_w_mean],
        [w_median, df1, df2, p_w_median],
        [_w_median, _df1, _df2, _p_w_median],
        [w_median_adj, df1, df2, p_w_median_adj]
    ]

    # 返回输出表格
    return output


def f_test_anova(row_values, col_values):
    new_row_values = []
    new_col_values = []
    col_values = [int(value) for value in col_values]
    for row, col in zip(row_values, col_values):
        if row != 0:
            new_row_values.append(row)
            new_col_values.append(col)
    # 计算总体均值
    total_mean = np.mean(new_row_values)

    # 计算组内平方和和自由度
    unique_factors = np.unique(new_col_values)
    num_factors = len(unique_factors)
    num_obs = len(new_row_values)

    within_sum_squares = 0.0
    within_degrees_freedom = num_obs - num_factors

    for factor in unique_factors:
        factor_indices = np.where(new_col_values == factor)[0]
        _row_values = []
        # 求组内均值
        for i in factor_indices:
            _row_values.append(new_row_values[i])
        _mean = np.mean(_row_values)

        for i in factor_indices:
            factor_values = new_row_values[i]
            within_sum_squares += np.sum((factor_values - _mean) ** 2)

    # 计算组间平方和和自由度
    between_sum_squares = np.sum((new_row_values - total_mean) ** 2) - within_sum_squares
    between_degrees_freedom = num_factors - 1
    # 计算组内和组间的均方
    mean_square_within = within_sum_squares / within_degrees_freedom
    mean_square_between = between_sum_squares / between_degrees_freedom

    between_sum_squares, within_sum_squares, mean_square_within, mean_square_between = \
        round(between_sum_squares, 3), \
            round(within_sum_squares, 3), \
            round(mean_square_within, 3), \
            round(mean_square_between, 3)
    # 计算F检验值和p值
    f_value = mean_square_between / mean_square_within
    p_value = 1 - stats.f.cdf(f_value, between_degrees_freedom, within_degrees_freedom)
    f_value, p_value = round(f_value, 3), round(p_value, 3)
    output = [
        [between_sum_squares, between_degrees_freedom, mean_square_between, f_value, p_value],
        [within_sum_squares, within_degrees_freedom, mean_square_within, '', ''],
        [between_sum_squares + within_sum_squares, between_degrees_freedom + within_degrees_freedom, '', '', '']
    ]
    return output


def chi_square_test(row_values, col_values):
    row_values = [int(value) for value in row_values]
    col_values = [int(value) for value in col_values]
    print(f'输入进卡方的组:\nrow_values{row_values}\ncol_values:{col_values}')
    print("row_values长度:", len(row_values))
    print("col_values长度:", len(col_values))
    print("col_values中的唯一值数量:", len(set(col_values)))
    print("row_values值的种类数量:", len(set(row_values)))
    observed = pd.crosstab(row_values, col_values)  # 计算观察到的频数
    chi2_val, p_val, dof, expected = stats.chi2_contingency(observed)  # 执行卡方检验
    # 计算连续性修正值和双侧渐进显著性.chi2_cont[0]表示连续性修正的值，chi2_cont[1]表示双侧渐进显著性（p 值）
    chi2_cont = np.sum((observed - expected - 0.5) ** 2 / expected)
    p_val_corr = 1 - stats.chi2.cdf(chi2_cont, df=(observed.shape[0] - 1) * (observed.shape[1] - 1))
    # chi2_cont_ =chi2_cont[0]
    # p_val_corr_ =p_val_corr[0]
    # 计算似然比和双侧渐进显著性
    likelihood_ratio, lr_p_val = stats.pearsonr(row_values, col_values)
    # # 计算费希尔精确检验的奇比值和双侧精确显著性
    # odds_ratio, exact_p_val = stats.fisher_exact(observed)
    # 计算线性关联
    linear_corr, linear_p_val = stats.linregress(row_values, col_values)[:2]
    # 计算有效个案数
    valid_cases = np.sum(observed)
    # 计算精确显著性(双侧)
    exact_p_val_two_sided = stats.chisquare(f_obs=observed, f_exp=expected, axis=None,
                                            ddof=observed.size - expected.size).pvalue
    # 计算精确显著性(单侧)
    exact_p_val_one_sided = exact_p_val_two_sided / 2

    # 构建结果字典
    result = {
        'Pearson Chi-square': {
            'Value': chi2_val,
            'Degrees of Freedom': dof,
            'Two-sided Asymptotic Significance': p_val,
            'Two-sided Exact Significance': '',
            'One-sided Exact Significance': ''
        },
        # 'Continuity Correction': {
        #     'Value': chi2_cont,
        #     'Degrees of Freedom': dof,
        #     'Two-sided Asymptotic Significance': p_val_corr,
        #     'Two-sided Exact Significance': '',
        #     'One-sided Exact Significance': ''
        # },
        'Likelihood Ratio': {
            'Value': likelihood_ratio,
            'Degrees of Freedom': dof,
            'Two-sided Asymptotic Significance': lr_p_val,
            'Two-sided Exact Significance': '',
            'One-sided Exact Significance': ''
        },
        # 'Fisher Exact Test': {
        #     'Value': odds_ratio,
        #     'Degrees of Freedom': '',
        #     'Two-sided Asymptotic Significance': '',
        #     'Two-sided Exact Significance': exact_p_val,
        #     'One-sided Exact Significance': exact_p_val / 2
        # },
        'Linear Correlation': {
            'Value': linear_corr,
            'Degrees of Freedom': dof,
            'Two-sided Asymptotic Significance': linear_p_val,
            'Two-sided Exact Significance': '',
            'One-sided Exact Significance': ''
        },
        'Valid Cases': {
            'Value': valid_cases.sum(),
            'Degrees of Freedom': '',
            'Two-sided Asymptotic Significance': '',
            'Two-sided Exact Significance': '',
            'One-sided Exact Significance': ''
        }
    }

    return result


def find_duplicates(arr):
    hash_table = {}
    duplicates = []

    # 遍历数组中的每个元素
    for num in arr:
        if num in hash_table:
            # 如果元素已经在哈希表中存在，则将其值加1
            hash_table[num] += 1
        else:
            # 如果元素不在哈希表中，则将其添加到哈希表中，值为1
            hash_table[num] = 1

    # # 遍历哈希表，找出值大于1的键
    # for key, value in hash_table.items():
    #     if value > 1:
    #         duplicates.append(key)

    return hash_table
