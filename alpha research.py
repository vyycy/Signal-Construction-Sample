import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed


def Alpha2(stock_data_high, stock_data_low, k):
    tempVal = ((stock_data_high.rolling(k).corr(stock_data_low)).rank(1)).rolling(k).sum()
    return tempVal


def Alpha3(stock_data_volume, k):
    tempVal = stock_data_volume
    return -tempVal.rolling(k).std()


def Alpha4(stock_data_volume, stock_data_high, k):
    tempVal1 = (stock_data_volume.rolling(k).cov(stock_data_high)).rank(1)
    tempVal2 = (stock_data_high.rolling(k).std()).rank(1)
    tempVal = -(tempVal1 * tempVal2)
    return tempVal


def Alpha5(stock_data_volume, stock_data_high, k):
    tempVal1 = ((stock_data_volume.rolling(k).cov(stock_data_high)).rank(1)).rolling(k).sum()
    tempVal2 = (stock_data_high.rolling(k).std()).rank(1)
    tempVal = -(tempVal1 * tempVal2)
    return tempVal


def Alpha6(stock_data_high, stock_data_low, stock_data_close, k):
    tempVal = (stock_data_high + stock_data_low) / stock_data_close
    return tempVal.rolling(k).sum()


def MktCap_process(x):
    """Process mktcap factos, eliminate outliers"""
    if (x.size == sum(x.isna())):
        return x
    dm = x - np.median(x[~np.isnan(x)])
    dm1 = np.median(abs(dm[~np.isnan(x)]))
    x[x > (dm + 5 * dm1)] = (dm + 5 * dm1)
    x[x < (dm - 5 * dm1)] = (dm - 5 * dm1)
    return x


def Factor_process(x, ind_mat, stock_mkt):
    """Process and normalize alpha factors"""
    if (x.size == sum(x.isna())):
        return x
    dm = x - np.median(x[~np.isnan(x)])
    dm1 = np.median(abs(dm[~np.isnan(x)]))
    x[x > (dm + 5 * dm1)] = (dm + 5 * dm1)
    x[x < (dm - 5 * dm1)] = (dm - 5 * dm1)

    dm = stock_mkt - np.median(stock_mkt[~np.isnan(stock_mkt)])
    dm1 = np.median(abs(dm[~np.isnan(stock_mkt)]))
    stock_mkt[stock_mkt > (dm + 5 * dm1)] = (dm + 5 * dm1)
    stock_mkt[stock_mkt < (dm - 5 * dm1)] = (dm - 5 * dm1)

    exo_var = np.append(ind_mat, [np.array(stock_mkt)], axis=0)

    model1 = sm.OLS(np.array(x), np.column_stack((exo_var)), missing='drop')
    results = model1.fit()

    x.loc[~((x * stock_mkt).isna())] = results.resid
    x.loc[((x * stock_mkt).isna())] = float("nan")

    x = (x - x.mean()) / x.std()
    return x


def SigTest(stock_return_period, stock_industry_sec, stock_signal_sec, stock_mktcap_sec, stock_isst_sec):
    """Alpha significant tests"""
    stock_return_period = stock_return_period[stock_isst_sec == 0]
    stock_signal_sec = stock_signal_sec[stock_isst_sec == 0]
    stock_mktcap_sec = stock_mktcap_sec[stock_isst_sec == 0]
    stock_industry_sec = [df[stock_isst_sec == 0] for df in stock_industry_sec]

    ret_var = np.array(stock_return_period)
    mktcap_var = np.array(np.log(stock_mktcap_sec))
    signal_var = np.array(stock_signal_sec)
    stock_industry_var = np.array(stock_industry_sec)

    exp_vars_temp = np.append([signal_var], stock_industry_var, axis=0)
    exp_vars = np.column_stack(np.append(exp_vars_temp, [mktcap_var], axis=0))
    end_var = ret_var
    weight_list = np.array(np.sqrt(stock_mktcap_sec))

    WLSmodel = sm.WLS(end_var, exp_vars, weights=weight_list, missing='drop')
    results = WLSmodel.fit()
    return results.tvalues[0]


def SigLevelTest(stock_isst_sec, stock_signal_value, stock_period_return):
    """Alphas' levels' return test"""
    stock_signal_value = stock_signal_value[stock_isst_sec == 0]
    stock_period_return = stock_period_return[stock_isst_sec == 0]

    tencile_cutpoint = np.array(stock_signal_value.quantile([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]))
    result_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for cutp in range(len(tencile_cutpoint) - 1):
        result_vec[cutp] = np.mean(stock_period_return[(stock_signal_value <= tencile_cutpoint[cutp]) & (
                stock_signal_value >= tencile_cutpoint[cutp + 1])])

    return result_vec


def SimTradingDouble(alpha2_trading_values, alpha3_trading_values, alpha4_trading_values, alpha5_trading_values,
                     alpha6_trading_values, stock_trading_st, stock_trading_return, num_stock):
    """Double side simulated simple trading result"""
    alpha2_trading_values = alpha2_trading_values[stock_trading_st == 0]
    alpha3_trading_values = alpha3_trading_values[stock_trading_st == 0]
    alpha4_trading_values = alpha4_trading_values[stock_trading_st == 0]
    alpha5_trading_values = alpha5_trading_values[stock_trading_st == 0]
    alpha6_trading_values = alpha6_trading_values[stock_trading_st == 0]

    alpha2_trading_thres = alpha2_trading_values.quantile(0.8)
    alpha3_trading_thres = alpha3_trading_values.quantile(0.7)
    alpha4_trading_thres = alpha4_trading_values.quantile(1)
    alpha5_trading_thres = alpha5_trading_values.quantile(1)
    alpha6_trading_thres = alpha6_trading_values.quantile(1)

    combined_trading_values = -0.5 * abs(alpha2_trading_values - alpha2_trading_thres) - 0.5 * abs(
        alpha3_trading_values - alpha3_trading_thres) - abs(
        alpha4_trading_values - alpha4_trading_thres) - abs(
        alpha5_trading_values - alpha5_trading_thres) - 0.7 * abs(alpha6_trading_values - alpha6_trading_thres)

    intended_position_long = combined_trading_values.sort_values(ascending=False).iloc[:num_stock].index
    stock_trading_return_selected_long = stock_trading_return[intended_position_long]

    intended_position_short = combined_trading_values.sort_values(ascending=True).iloc[:num_stock].index
    stock_trading_return_selected_short = stock_trading_return[intended_position_short]

    period_port_return_long = stock_trading_return_selected_long.mean(axis=1)
    period_port_return_short = -stock_trading_return_selected_short.mean(axis=1)
    return pd.concat([period_port_return_long, period_port_return_short], axis=1), np.array(intended_position_long)


#read and process dataframes
return_data = pd.read_csv('data/return.csv')
stock_data = pd.read_csv('data/stock.csv')
stock_citics_data = pd.read_csv('data/stock_citics.csv')
stock_sw_data = pd.read_csv('data/stock_sw.csv')
stock_data['Return'] = stock_data['Close'] / stock_data['PreClose'] - 1

stock_data_high = stock_data[['TradeDate', 'Uid', 'High']].pivot(index='TradeDate', columns='Uid', values='High')
stock_data_low = stock_data[['TradeDate', 'Uid', 'Low']].pivot(index='TradeDate', columns='Uid', values='Low')
stock_data_close = stock_data[['TradeDate', 'Uid', 'Close']].pivot(index='TradeDate', columns='Uid', values='Close')
stock_data_volume = stock_data[['TradeDate', 'Uid', 'Volume']].pivot(index='TradeDate', columns='Uid', values='Volume')
stock_data_return = stock_data[['TradeDate', 'Uid', 'Return']].pivot(index='TradeDate', columns='Uid', values='Return')
stock_data_mktcap = stock_data[['TradeDate', 'Uid', 'MarketCap']].pivot(index='TradeDate', columns='Uid',
                                                                        values='MarketCap')
stock_data_st = stock_data[['TradeDate', 'Uid', 'IsSt']].pivot(index='TradeDate', columns='Uid', values='IsSt')

alpha2_mat = Alpha2(stock_data_high, stock_data_low, 20)
alpha3_mat = Alpha3(stock_data_volume, 5)
alpha4_mat = Alpha4(stock_data_volume, stock_data_high, 10)
alpha5_mat = Alpha5(stock_data_volume, stock_data_high, 5)
alpha6_mat = Alpha6(stock_data_high, stock_data_low, stock_data_close, 5)

#generate industry factor vectors
stock_data_industry = stock_citics_data[['Date', 'Uid', 'INDUSTRIESCODEL1']].pivot(index='Date', columns='Uid',
                                                                                   values='INDUSTRIESCODEL1')
unique_ind_code = pd.unique(stock_citics_data['INDUSTRIESCODEL1'])
stock_data_industry = (stock_data_industry.combine(stock_data_close, lambda s1, s2: s1))[stock_data_close.columns]

ind_data_return = pd.DataFrame()

stock_data_indmat_list = []

for indCode in unique_ind_code:
    stock_data_indmat_list.append((stock_data_industry.eq(indCode)) * 1)

stock_data_mktcap = stock_data_mktcap.apply(lambda x: MktCap_process(x), axis=1)


#Process alpha factors for later testing
alpha2_mat_process = alpha2_mat.apply(
    lambda x: Factor_process(x, np.array([df.loc[x.name] for df in stock_data_indmat_list]),
                             stock_data_mktcap.loc[x.name]), axis=1)
alpha3_mat_process = alpha3_mat.apply(
    lambda x: Factor_process(x, np.array([df.loc[x.name] for df in stock_data_indmat_list]),
                             stock_data_mktcap.loc[x.name]), axis=1)
alpha4_mat_process = alpha4_mat.apply(
    lambda x: Factor_process(x, np.array([df.loc[x.name] for df in stock_data_indmat_list]),
                             stock_data_mktcap.loc[x.name]), axis=1)
alpha5_mat_process = alpha5_mat.apply(
    lambda x: Factor_process(x, np.array([df.loc[x.name] for df in stock_data_indmat_list]),
                             stock_data_mktcap.loc[x.name]), axis=1)
alpha6_mat_process = alpha6_mat.apply(
    lambda x: Factor_process(x, np.array([df.loc[x.name] for df in stock_data_indmat_list]),
                             stock_data_mktcap.loc[x.name]), axis=1)

stock_data_industry = stock_data_industry.fillna('None')

period = 20
stock_data_return_cumul = np.exp(np.log(stock_data_return + 1).rolling(period).sum()) - 1

date_vec = np.array(stock_data_return_cumul.index)

ind_begin = 60
ind_end = 1210

#test alphas' significant levels
alpha2_tvalues = Parallel(n_jobs=-1)(
    delayed(SigTest)(stock_data_return_cumul.loc[date_vec[begin + period]],
                     [df.loc[date_vec[begin]] for df in stock_data_indmat_list],
                     alpha2_mat_process.loc[date_vec[begin]],
                     stock_data_mktcap.loc[date_vec[begin]],
                     stock_data_st.loc[date_vec[begin]]) for begin in range(ind_begin, ind_end))
alpha2_tvalues_abs_mean = np.mean(abs(np.array(alpha2_tvalues)))
alpha2_tvalues_big2 = sum(abs(np.array(alpha2_tvalues)) > 2) / len(alpha2_tvalues)
alpha2_tvalues_mean = np.mean(alpha2_tvalues)
alpha2_tvalues_chrac_vec = [alpha2_tvalues_abs_mean, alpha2_tvalues_big2, alpha2_tvalues_mean]

alpha3_tvalues = Parallel(n_jobs=-1)(
    delayed(SigTest)(stock_data_return_cumul.loc[date_vec[begin + period]],
                     [df.loc[date_vec[begin]] for df in stock_data_indmat_list],
                     alpha3_mat_process.loc[date_vec[begin]],
                     stock_data_mktcap.loc[date_vec[begin]],
                     stock_data_st.loc[date_vec[begin]]) for begin in range(ind_begin, ind_end))
alpha3_tvalues_abs_mean = np.mean(abs(np.array(alpha3_tvalues)))
alpha3_tvalues_big2 = sum(abs(np.array(alpha3_tvalues)) > 2) / len(alpha3_tvalues)
alpha3_tvalues_mean = np.mean(alpha3_tvalues)
alpha3_tvalues_chrac_vec = [alpha3_tvalues_abs_mean, alpha3_tvalues_big2, alpha3_tvalues_mean]

alpha4_tvalues = Parallel(n_jobs=-1)(
    delayed(SigTest)(stock_data_return_cumul.loc[date_vec[begin + period]],
                     [df.loc[date_vec[begin]] for df in stock_data_indmat_list],
                     alpha4_mat_process.loc[date_vec[begin]],
                     stock_data_mktcap.loc[date_vec[begin]],
                     stock_data_st.loc[date_vec[begin]]) for begin in range(ind_begin, ind_end))
alpha4_tvalues_abs_mean = np.mean(abs(np.array(alpha4_tvalues)))
alpha4_tvalues_big2 = sum(abs(np.array(alpha4_tvalues)) > 2) / len(alpha4_tvalues)
alpha4_tvalues_mean = np.mean(alpha4_tvalues)
alpha4_tvalues_chrac_vec = [alpha4_tvalues_abs_mean, alpha4_tvalues_big2, alpha4_tvalues_mean]

alpha5_tvalues = Parallel(n_jobs=-1)(
    delayed(SigTest)(stock_data_return_cumul.loc[date_vec[begin + period]],
                     [df.loc[date_vec[begin]] for df in stock_data_indmat_list],
                     alpha5_mat_process.loc[date_vec[begin]],
                     stock_data_mktcap.loc[date_vec[begin]],
                     stock_data_st.loc[date_vec[begin]]) for begin in range(ind_begin, ind_end))
alpha5_tvalues_abs_mean = np.mean(abs(np.array(alpha5_tvalues)))
alpha5_tvalues_big2 = sum(abs(np.array(alpha5_tvalues)) > 2) / len(alpha5_tvalues)
alpha5_tvalues_mean = np.mean(alpha5_tvalues)
alpha5_tvalues_chrac_vec = [alpha5_tvalues_abs_mean, alpha5_tvalues_big2, alpha5_tvalues_mean]

alpha6_tvalues = Parallel(n_jobs=-1)(
    delayed(SigTest)(stock_data_return_cumul.loc[date_vec[begin + period]],
                     [df.loc[date_vec[begin]] for df in stock_data_indmat_list],
                     alpha6_mat_process.loc[date_vec[begin]],
                     stock_data_mktcap.loc[date_vec[begin]],
                     stock_data_st.loc[date_vec[begin]]) for begin in range(ind_begin, ind_end))
alpha6_tvalues_abs_mean = np.mean(abs(np.array(alpha6_tvalues)))
alpha6_tvalues_big2 = sum(abs(np.array(alpha6_tvalues)) > 2) / len(alpha6_tvalues)
alpha6_tvalues_mean = np.mean(alpha6_tvalues)
alpha6_tvalues_chrac_vec = [alpha6_tvalues_abs_mean, alpha6_tvalues_big2, alpha6_tvalues_mean]

#Summarize alpha test t-values' result
alphas_charact = pd.DataFrame(np.column_stack(
    [alpha2_tvalues_chrac_vec, alpha3_tvalues_chrac_vec, alpha4_tvalues_chrac_vec, alpha5_tvalues_chrac_vec,
     alpha6_tvalues_chrac_vec]), columns=['Alpha2', 'Alpha3', 'Alpha4', 'Alpha5', 'Alpha6'],
    index=['abs_mean', 'abs ratio >2', 'mean'])

#Perform level return test for each alpha
alpha2_levels = Parallel(n_jobs=-1)(
    delayed(SigLevelTest)(stock_data_st.loc[date_vec[date_ind]], alpha2_mat_process.loc[date_vec[date_ind]],
                          stock_data_return_cumul.loc[date_vec[date_ind + period]]) for date_ind in
    range(ind_begin, ind_end))
alpha2_levels_mean = np.mean(alpha2_levels, axis=0)

alpha3_levels = Parallel(n_jobs=-1)(
    delayed(SigLevelTest)(stock_data_st.loc[date_vec[date_ind]], alpha3_mat_process.loc[date_vec[date_ind]],
                          stock_data_return_cumul.loc[date_vec[date_ind + period]]) for date_ind in
    range(ind_begin, ind_end))
alpha3_levels_mean = np.mean(alpha3_levels, axis=0)

alpha4_levels = Parallel(n_jobs=-1)(
    delayed(SigLevelTest)(stock_data_st.loc[date_vec[date_ind]], alpha4_mat_process.loc[date_vec[date_ind]],
                          stock_data_return_cumul.loc[date_vec[date_ind + period]]) for date_ind in
    range(ind_begin, ind_end))
alpha4_levels_mean = np.mean(alpha4_levels, axis=0)

alpha5_levels = Parallel(n_jobs=-1)(
    delayed(SigLevelTest)(stock_data_st.loc[date_vec[date_ind]], alpha5_mat_process.loc[date_vec[date_ind]],
                          stock_data_return_cumul.loc[date_vec[date_ind + period]]) for date_ind in
    range(ind_begin, ind_end))
alpha5_levels_mean = np.mean(alpha5_levels, axis=0)

alpha6_levels = Parallel(n_jobs=-1)(
    delayed(SigLevelTest)(stock_data_st.loc[date_vec[date_ind]], alpha6_mat_process.loc[date_vec[date_ind]],
                          stock_data_return_cumul.loc[date_vec[date_ind + period]]) for date_ind in
    range(ind_begin, ind_end))
alpha6_levels_mean = np.mean(alpha6_levels, axis=0)

level_result_combine = pd.DataFrame(
    data=np.transpose([alpha2_levels_mean, alpha3_levels_mean, alpha4_levels_mean, alpha5_levels_mean,
                       alpha6_levels_mean]), index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    columns=['Alpha2', 'Alpha3', 'Alpha4', 'Alpha5', 'Alpha6'])


#Simulating trade process and calculate statistics
trade_begin_ind = 1214
trade_end_ind = 1934
num_stock = 65
trade_period = 10

sim_trade_result, intend_positions = zip(*Parallel(n_jobs=-1)(
    delayed(SimTradingDouble)(alpha2_mat_process.loc[date_vec[ind_tradedate]],
                              alpha3_mat_process.loc[date_vec[ind_tradedate]],
                              alpha4_mat_process.loc[date_vec[ind_tradedate]],
                              alpha5_mat_process.loc[date_vec[ind_tradedate]],
                              alpha6_mat_process.loc[date_vec[ind_tradedate]],
                              stock_data_st.loc[date_vec[ind_tradedate]],
                              stock_data_return.loc[date_vec[(ind_tradedate + 1):(ind_tradedate + 1 + trade_period)]],
                              num_stock) for
    ind_tradedate in
    range(trade_begin_ind, (trade_end_ind + 1), trade_period)))

sim_trade_result_concate = pd.concat(sim_trade_result)
sim_trade_result_concate_cumsum = sim_trade_result_concate.cumsum()

ssindex_data = pd.read_csv('data/SS000001.csv')
ssindex_data.index = ssindex_data['Date']
ssindex_data_close = ssindex_data['Adj Close']
ssindex_data_close_return = ssindex_data_close.pct_change()
ssindex_data_close_return_match = ssindex_data_close_return[sim_trade_result_concate.index]
ssindex_return_match_cumsum = ssindex_data_close_return_match.cumsum()

excess_return = sim_trade_result_concate[0] - ssindex_data_close_return_match
excess_return_cumprod = (excess_return + 1).cumprod() - 1

turnover_ratio = []
for i in range(1, len(intend_positions)):
    turnover_ratio.append(
        len(set(np.array(intend_positions)[i]).difference(set(np.array(intend_positions)[i - 1]))) / num_stock)
turnover_ratio_mean = np.mean(turnover_ratio)
Sharpe_r = np.sqrt(240) * np.mean(excess_return) / np.std(excess_return)

excess_return_cumprod.plot()
