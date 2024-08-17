import pandas as pd
import numpy as np
import os
import sys
import torch
from itertools import cycle
import torch.nn.functional as F
import torch.nn as nn
import copy
import glob
import re
import DenseWideNet
from collections import OrderedDict
from scipy import stats
import matplotlib.pyplot as plt
import datashader as ds
from datashader.mpl_ext import dsshow
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AverageModels:
    def __init__(self):
        self.saved_state_dic = []

    def update(self, state_dic):
        self.saved_state_dic.append(state_dic)

    def average_state_dic(self):
        saveNum = len(self.saved_state_dic)
        temp = copy.deepcopy(self.saved_state_dic[0])
        for key in self.saved_state_dic[0]:
            for i in range(1, saveNum):
                temp[key] = temp[key] + self.saved_state_dic[i][key]
            temp[key] = temp[key] / saveNum
        return temp


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunningLoss:
    def __init__(self, domain_losstype, mv_losstype, domain_index=None):
        if domain_index is None:
            domain_index = [3]
        self.joint_loss = AverageMeter()
        self.fmv_loss = AverageMeter()
        self.cmv_loss = AverageMeter()
        self.transfer_loss = AverageMeter()
        self.domain_losstype = domain_losstype
        self.mv_losstype = mv_losstype
        self.domain_index=domain_index

    def cal_transfer_loss(self, source_fe, target_fe):
        if self.domain_losstype == "mmd":
            MMD = MMD_loss(kernel_type='rbf', kernel_mul=2.0, kernel_num=5)  # kernel_type='linear'
            loss = MMD.forward(source_fe, target_fe)
            #loss = loss ** 2
        elif self.domain_losstype == "mmdl":
            loss = self.mmd_linear(source_fe, target_fe)
        elif self.domain_losstype == 'coral':
            loss = CORAL(source_fe, target_fe)
            loss = torch.sqrt(loss)
        else:
            print("WARNING: No valid transfer loss function is used.")
            loss = 0
        return loss

    def cal_mtransfer_loss(self, fcs, source_size, temp_size):
        transfer_loss=[]
        for i in self.domain_index:
            source_fe, target_fe = fcs[i].narrow(0, 0, temp_size), fcs[i].narrow(0, source_size, temp_size)
            loss= self.cal_transfer_loss(source_fe, target_fe)
            transfer_loss.append(loss)
        transfer_loss = sum(transfer_loss) / len(transfer_loss)
        return transfer_loss

    def cal_mtransfer_loss_v2(self, fcs, source_size, target_size):
        transfer_loss=[]
        for i in self.domain_index:
            source_fe, target_fe = fcs[i].narrow(0, 0, source_size), fcs[i].narrow(0, source_size, target_size)
            for s in range(50):
                tempf, _ = train_test_split(target_fe, train_size=source_size, random_state=s, shuffle=True)
                loss = self.cal_transfer_loss(source_fe, target_fe)
                transfer_loss.append(loss)
        transfer_loss = sum(transfer_loss) / len(transfer_loss)
        return transfer_loss

    def cal_mv_loss(self, y_pre, y_true):
        if self.mv_losstype == 'MSE':
            loss = F.mse_loss(y_pre, y_true)
        elif self.mv_losstype == 'MAPE':
            loss = self.MAPE(y_pre,y_true)
        else:
            print("WARNING: No valid soil moisture loss function is used.")
            loss = 0
        return loss

    def input4epoch(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'source':
                self.source_x, self.source_y = value
            elif key == 'target':
                self.target_x, self.target_y = value
            elif key == 'coarse':
                self.coarse_x, self.coarse_y = value
            else:
                print("WARNING: key error.")
                sys.exit(1)

    def epoch_loss(self):
        return [self.joint_loss.avg, self.fmv_loss.avg, self.transfer_loss.avg, self.cmv_loss.avg]

    def MAPE(self, y_pre, y_true):
        loss = torch.mean(torch.abs((y_pre - y_true) / y_true))
        return loss

    def mmd_linear(self, f_of_X, f_of_Y):
        delta = f_of_X - f_of_Y
        loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
        return loss


class ResultsAnalysis:
    def __init__(self, y_true, y_pred):
        if y_true.dtype == 'float32' or y_true.dtype == 'float64':
            self.y_true = y_true
        else:
            self.y_true = y_true.cpu().squeeze().numpy()
        if y_pred.dtype == 'float32' or y_pred.dtype == 'float64':
            self.y_pred = y_pred
        else:
            self.y_pred = y_pred.cpu().squeeze().numpy()
        self._calc_stastics()
        self._fitted_line()

    def _calc_stastics(self):
        bias = np.mean(self.y_pred) - np.mean(self.y_true)
        rmse = np.sqrt(np.mean(np.square(self.y_pred - self.y_true)))
        r = np.corrcoef(self.y_true, self.y_pred)
        ubrmse = np.sqrt(np.mean(np.square(self.y_pred - self.y_true - bias)))
        invR = 1 - np.sum((self.y_pred < 0) | (self.y_pred > 0.6)) / self.y_pred.shape[0]
        self.stat = np.asarray([bias, r[0, 1], rmse, ubrmse, invR])
        self.stat_3d = np.round(self.stat * 1000) / 1000

    def _fitted_line(self):
        slope, intercept, r_value, p_value, std_err = stats.linregress(self.y_true, self.y_pred)
        self.fitted_line = [slope, intercept, r_value, p_value, std_err]

    def scatter_density_fig(self, fig_dir=''):
        bins = [1000, 1000]  # number of bins
        # histogram the data
        hh, locx, locy = np.histogram2d(self.y_true, self.y_pred, bins=bins)
        linex = np.asarray(list(range(2, 50))) / 100
        # Sort the points by density, so that the densest points are plotted last
        z = np.array([hh[np.argmax(a <= locx[1:]), np.argmax(b <= locy[1:])] for a, b in zip(self.y_true, self.y_pred)])
        idx = z.argsort()
        x2, y2, z2 = self.y_true[idx], self.y_pred[idx], z[idx]
        plt.figure(1, figsize=(5, 5)).clf()
        plt.text(0.05, 0.65, 'Bias=' + str(self.stat_3d[0]), size=10)
        plt.text(0.05, 0.62, 'R=' + str(self.stat_3d[1]), size=10)
        plt.text(0.05, 0.59, 'RMSE=' + str(self.stat_3d[2]), size=10)
        plt.text(0.05, 0.56, 'ubRMSE=' + str(self.stat_3d[3]), size=10)
        plt.text(0.05, 0.53, 'InvRate=' + str(self.stat_3d[4]), size=10)
        plt.ylabel('Predicted[m$^3$/m$^3$]', fontsize=12)
        plt.xlabel('Measured[m$^3$/m$^3$]', fontsize=12)
        s = plt.scatter(x2, y2, c=z2, s=2, cmap='jet', marker='.')

        plt.plot(linex, linex * self.fitted_line[0] + self.fitted_line[1], color='red')
        plt.plot(np.arange(0, 0.8, 0.1), np.arange(0, 0.8, 0.1), '--k')
        slope = str(np.round(self.fitted_line[0] * 1000) / 1000)
        intercept = str(np.round(self.fitted_line[1] * 1000) / 1000)
        plt.text(0.4, 0.05, 'y=%sx+%s' % (slope, intercept), size=12, color=[1, 0, 0])
        plt.axis([0, 0.7, 0, 0.7])
        if fig_dir:
            plt.savefig(fig_dir, dpi=300)

    def using_datashader(self, fname, vm, sc, fig_dir=''):
        df = pd.DataFrame(dict(x=self.y_true, y=self.y_pred))
        fig, ax = plt.subplots(figsize=(6, 5))
        dsartist = dsshow(
            df,
            ds.Point("x", "y"),
            ds.count(),
            vmin=0,
            vmax=vm,
            cmap='viridis',  # "jet"'viridis',  ['darkgreen','yellow'],
            norm="linear",
            aspect="auto",
            width_scale=sc,
            height_scale=sc,
            ax=ax
        )
        linex = np.asarray(list(range(10, 40))) / 100
        # Sort the points by density, so that the densest points are plotted last
        plt.text(0.05, 0.55, 'Bias=' + str(self.stat_3d[0]), size=14, color=[0, 0, 0])
        plt.text(0.05, 0.515, 'R=' + str(self.stat_3d[1]), size=14, color=[0, 0, 0])
        plt.text(0.3, 0.55, 'RMSE=' + str(self.stat_3d[2]), size=14, color=[0, 0, 0])
        plt.text(0.3, 0.515, 'ubRMSE=' + str(self.stat_3d[3]), size=14, color=[0, 0, 0])
        # plt.text(0.05, 0.51, '#samples=' + str(self.y_true.shape[0]), size=12, color=[1, 0, 0])
        plt.ylabel('Predicted[m$^3$/m$^3$]', fontsize=14)
        plt.xlabel('Measured[m$^3$/m$^3$]', fontsize=14)

        plt.plot(linex, linex * self.fitted_line[0] + self.fitted_line[1], color='red')
        plt.plot(np.arange(0, 0.8, 0.1), np.arange(0, 0.8, 0.1), '--k')
        slope = str(np.round(self.fitted_line[0] * 1000) / 1000)
        intercept = str(np.round(self.fitted_line[1] * 1000) / 1000)
        plt.text(0.3, 0.07, 'y=%sx+%s' % (slope, intercept), size=14, color=[1, 0, 0])
        plt.text(0.3, 0.03, fname, size=14, color=[0, 0, 0])
        plt.axis([0, 0.6, 0, 0.6])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.colorbar(dsartist)
        plt.tight_layout()
        if fig_dir:
            fig.savefig(fig_dir, dpi=300)


class TrainHist:
    def __init__(self):
        self.trainLoss = []
        self.valLoss = []

    def update(self, trainLoss, valLoss):
        self.trainLoss.append(trainLoss)
        self.valLoss.append(valLoss)

    def report_hist(self):
        columns = ['jloss', 'fmv', 'domain', 'cmv']
        trainLoss = pd.DataFrame(self.trainLoss, columns=columns)
        valLoss = pd.DataFrame(self.valLoss, columns=['Val_Bias', 'Val_R', 'Val_RMSE', 'Val_ubRMSE', 'Inv_Rate'])
        hist_table = pd.concat([trainLoss, valLoss], axis=1)
        hist_table.index.name = 'epoch'
        return hist_table


class FinetuneModel:
    def __init__(self, setup, data):
        self.lr = setup['lr']
        self.epoch_Num = setup['epoch_Num']
        self.swa_start = setup['swa_start']
        self.alpha = setup['alpha']
        self.beta = setup['beta']
        self.domain_type = setup['domain_type']
        self.mv_type = setup['mv_type']
        self.val_data = data['val_data']
        self.tr_data = data['train_data']
        self.sl = data['sl']
        self.tl = data['tl']
        self.cl = data['cl']
        self.m = setup['ex'][:4]

    def cal_beta(self, model):
        model.fe.to(device)
        model.mv.to(device)
        run_loss = RunningLoss(self.domain_type, self.mv_type)
        with torch.no_grad():
            model.fe.eval()
            model.mv.eval()

            source_x, source_y = self.tr_data
            source_x, source_y = source_x.to(device), source_y.to(device)
            target_x, target_y = self.val_data
            target_x = target_x.to(device)

            inputs = torch.cat((source_x, target_x), dim=0)
            fe = model.fe(inputs)
            mv = model.mv(fe)
            source_size = source_y.shape[0]
            target_size = target_x.shape[0]
            source_fe, target_fe = fe.narrow(0, 0, source_size), fe.narrow(0, source_size, target_size)
            transfer_loss=[]
            for s in range(50):
                temp_fe,_ = train_test_split(target_fe, train_size=source_size, random_state=s, shuffle=True)
                temp_loss = run_loss.cal_transfer_loss(source_fe, temp_fe)
                transfer_loss.append(temp_loss)

            fmv_loss = run_loss.cal_mv_loss(mv.narrow(0, 0, source_size), source_y)
            transfer_loss = sum(transfer_loss)/len(transfer_loss)
            beta = fmv_loss * self.alpha/transfer_loss #eq.4
            #if beta > 500:
            #    beta = 500
            beta = beta.detach().cpu()
        return beta

    def cal_beta_v2(self, model):
        model.to(device)
        run_loss = RunningLoss(self.domain_type, self.mv_type)
        with torch.no_grad():
            model.eval()
            source_x, source_y = self.tr_data
            source_x, source_y = source_x.to(device), source_y.to(device)
            target_x, target_y = self.val_data
            target_x = target_x.to(device)
            inputs = torch.cat((source_x, target_x), dim=0)
            mv, fc1, fc2, fc3, fc4 = model(inputs)
            source_size = source_y.shape[0]
            target_size = target_x.shape[0]

            transfer_loss = run_loss.cal_mtransfer_loss_v2([fc1, fc2, fc3, fc4], source_size, target_size)

            fmv_loss = run_loss.cal_mv_loss(mv.narrow(0, 0, source_size), source_y)

            beta = fmv_loss * self.alpha*2.5/transfer_loss

            beta = beta.detach().cpu()
        return beta

    def ft3(self, model):
        if self.beta == 'auto':
            beta = self.cal_beta(model)
        else:
            beta = self.beta


        model.fe.to(device)
        model.mv.to(device)
        optimizer_fe = torch.optim.Adam(model.fe.parameters(), lr=self.lr)
        optimizer_mv = torch.optim.Adam(model.mv.parameters(), lr=self.lr)
        average_state = AverageModels()
        hist = TrainHist()

        xtest, ytest = self.val_data
        for epoch in range(self.epoch_Num):
            run_loss = RunningLoss(self.domain_type, self.mv_type)
            model.fe.train()
            model.mv.train()
            for item1, item2, item3 in zip(self.sl, cycle(self.tl), cycle(self.cl)):
                source_x, source_y = item1
                source_x, source_y = source_x.to(device), source_y.to(device)
                target_x, target_y = item2
                target_x = target_x.to(device)
                coarse_x, coarse_y = item3
                coarse_x, coarse_y = coarse_x.to(device), coarse_y.to(device)
                inputs = torch.cat((source_x, target_x, coarse_x), dim=0)
                optimizer_fe.zero_grad()
                optimizer_mv.zero_grad()
                fe = model.fe(inputs)
                mv = model.mv(fe)

                source_size = source_y.shape[0]
                target_size = target_x.shape[0]
                coarse_size = coarse_y.shape[0]
                temp_batch_size = min(source_size, target_size)
                source_fe, target_fe = fe.narrow(0, 0, temp_batch_size), fe.narrow(0, source_size, temp_batch_size)
                transfer_loss = run_loss.cal_transfer_loss(source_fe, target_fe)
                fmv_loss = run_loss.cal_mv_loss(mv.narrow(0, 0, source_size), source_y)
                cmv_loss = run_loss.cal_mv_loss(mv.narrow(0, source_size + target_size, coarse_size), coarse_y)
                loss = fmv_loss * self.alpha + cmv_loss * (1 - self.alpha) + transfer_loss * beta

                run_loss.joint_loss.update(loss.detach().cpu().numpy())
                run_loss.transfer_loss.update(transfer_loss.detach().cpu().numpy())
                run_loss.fmv_loss.update(fmv_loss.detach().cpu().numpy())
                run_loss.cmv_loss.update(cmv_loss.detach().cpu().numpy())

                loss.backward()
                optimizer_fe.step()
                optimizer_mv.step()

            with torch.no_grad():
                model.eval()
                y_pred = model(xtest)
                res = ResultsAnalysis(ytest, y_pred)

            hist.update(run_loss.epoch_loss(), res.stat)

            if epoch >= self.swa_start:
                average_state.update(copy.deepcopy(model.state_dict()))
        # res.scatter_density_fig()
        average_model_state = average_state.average_state_dic()
        return average_model_state, hist.report_hist(), res

    def ft2(self, model):
        if self.beta == 'auto':
            beta = self.cal_beta(model)
            #print(beta)
        else:
            beta = self.beta
        model.fe.to(device)
        model.mv.to(device)
        optimizer_fe = torch.optim.Adam(model.fe.parameters(), lr=self.lr)
        optimizer_mv = torch.optim.Adam(model.mv.parameters(), lr=self.lr)
        average_state = AverageModels()
        hist = TrainHist()

        xtest, ytest = self.val_data
        for epoch in range(self.epoch_Num):
            run_loss = RunningLoss(self.domain_type, self.mv_type)
            model.fe.train()
            model.mv.train()
            for item1, item2 in zip(self.sl, cycle(self.tl)):
                source_x, source_y = item1
                source_x, source_y = source_x.to(device), source_y.to(device)
                target_x, target_y = item2
                target_x = target_x.to(device)

                inputs = torch.cat((source_x, target_x), dim=0)
                optimizer_fe.zero_grad()
                optimizer_mv.zero_grad()
                fe = model.fe(inputs)
                mv = model.mv(fe)

                source_size = source_y.shape[0]
                target_size = target_x.shape[0]
                temp_batch_size = min(source_size, target_size)
                source_fe, target_fe = fe.narrow(0, 0, temp_batch_size), fe.narrow(0, source_size, temp_batch_size)
                transfer_loss = run_loss.cal_transfer_loss(source_fe, target_fe)
                fmv_loss = run_loss.cal_mv_loss(mv.narrow(0, 0, source_size), source_y)

                loss = fmv_loss * self.alpha + transfer_loss * beta/2.5

                run_loss.joint_loss.update(loss.detach().cpu().numpy())
                run_loss.transfer_loss.update(transfer_loss.detach().cpu().numpy())
                run_loss.fmv_loss.update(fmv_loss.detach().cpu().numpy())

                loss.backward()
                optimizer_fe.step()
                optimizer_mv.step()

            with torch.no_grad():
                model.eval()
                y_pred = model(xtest)
                res = ResultsAnalysis(ytest, y_pred)

            hist.update(run_loss.epoch_loss(), res.stat)

            if epoch >= self.swa_start:
                average_state.update(copy.deepcopy(model.state_dict()))
        # res.scatter_density_fig()
        average_model_state = average_state.average_state_dic()
        return average_model_state, hist.report_hist(), res

    def ft2_v2(self, model):
        if len(self.tr_data)>0:
            beta = self.cal_beta_v2(model)
            #print(beta)
        else:
            beta = self.beta
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        average_state = AverageModels()
        hist = TrainHist()
        xtest, ytest = self.val_data
        for epoch in range(self.epoch_Num):
            run_loss = RunningLoss(self.domain_type, self.mv_type)
            model.train()
            for item1, item2 in zip(self.sl, cycle(self.tl)):
                source_x, source_y = item1
                source_x, source_y = source_x.to(device), source_y.to(device)
                target_x, target_y = item2
                target_x = target_x.to(device)
                inputs = torch.cat((source_x, target_x), dim=0)
                optimizer.zero_grad()

                mv, tfc, fc1, fc2, fc3 = model(inputs)

                source_size = source_y.shape[0]
                target_size = target_x.shape[0]
                temp_size = min(source_size, target_size)
                transfer_loss = run_loss.cal_mtransfer_loss([tfc,fc1,fc2,fc3], source_size, temp_size)
                fmv_loss = run_loss.cal_mv_loss(mv.narrow(0, 0, source_size), source_y)

                loss = fmv_loss * self.alpha + transfer_loss * beta / 2.5

                run_loss.joint_loss.update(loss.detach().cpu().numpy())
                run_loss.transfer_loss.update(transfer_loss.detach().cpu().numpy())
                run_loss.fmv_loss.update(fmv_loss.detach().cpu().numpy())

                loss.backward()
                optimizer.step()


            with torch.no_grad():
                model.eval()
                y_pred = model(xtest)
                if len(y_pred)>1:
                    y_pred=y_pred[0]
                res = ResultsAnalysis(ytest, y_pred)


            hist.update(run_loss.epoch_loss(), res.stat)

            if epoch >= self.swa_start:
                average_state.update(copy.deepcopy(model.state_dict()))
        # res.scatter_density_fig()
        average_model_state = average_state.average_state_dic()
        return average_model_state, hist.report_hist(), res

    def domain_loss_single(self,source_size,temp_size,fc,run_loss):
        source_fe, target_fe = fc.narrow(0, 0, temp_size), fc.narrow(0, source_size, temp_size)
        transfer_loss = run_loss.cal_transfer_loss(source_fe, target_fe)
        return transfer_loss


    def ft_ensemble(self, modelX, model_dir):

        for i in range(len(modelX.DenseSME)):  # len(modelX.DenseSME)
            #print(f'Finetune {modelX.m_name[i]}')
            path_2_ave_model = os.path.join(model_dir, '%s.pt' % modelX.m_name[i])
            if os.path.exists(path_2_ave_model):
                continue
            path_2_fig = os.path.join(model_dir, '%s.jpg' % modelX.m_name[i])
            path_2_hist = os.path.join(model_dir, '%s.csv' % modelX.m_name[i])

            if self.m == 'ft13':
                ave_state, df_hist, res = self.ft3(modelX.DenseSME[i])
            elif self.m =='ft12':
                ave_state, df_hist, res = self.ft2(modelX.DenseSME[i])
            elif self.m == 'ft22':
                ave_state, df_hist, res = self.ft2_v2(modelX.DenseSME[i])

            res.scatter_density_fig(path_2_fig)  # need to reproduce
            df_hist.to_csv(path_2_hist)
            torch.save(ave_state, path_2_ave_model)



class Build_DenseSM:
    def __init__(self, arc_p):
        self.arc_p = arc_p
        self.m_name = []
        self.model_state = []
        self.parse_arc()
        self.DenseSME = []
        self.y_pred = []
        self.y_true = []

    def parse_arc(self):
        if isinstance(self.arc_p, str):
            pre_model_list = glob.glob(os.path.join(self.arc_p, '*.pt'))
            pre_model_list = sorted(pre_model_list, key=self.extract_sort_keys)
            arc_p = []
            # print(pre_model_list)
            for pre_model in pre_model_list:
                temp_arc_p = re.search('\d+_\d+', pre_model).group().split('_')
                self.m_name.append('m_%s_%s' % tuple(temp_arc_p))
                temp_arc_p = [int(p) for p in temp_arc_p]
                arc_p.append(temp_arc_p)
                temp_state = torch.load(pre_model)
                self._modify_keys(temp_state)
                self.model_state.append(temp_state)
            self.arc_p = arc_p
        else:
            for temp_arc_p in list(self.arc_p):
                temp_arc_p = [str(p) for p in temp_arc_p]
                self.m_name.append('m_%s_%s' % tuple(temp_arc_p))
            self.arc_p = [list(temp) for temp in list(self.arc_p)]


    def build_DenseSME(self):
        for temp_arc_p in self.arc_p:
            bk, wd = list(temp_arc_p)
            self.DenseSME.append(self.build_single(wd, bk))

    def build_single(self, wd, bk, lnum=1, fcNum=32):
        model = nn.Sequential(OrderedDict([
            ('fe', DenseWideNet.DWN_feature(wd, bk, lnum)),
            ('mv', nn.Linear(fcNum, 1)),
        ]))
        return model

    def rebuild_DenseSME(self, freezeFe=False):
        for temp_arc_p, temp_model_state in zip(self.arc_p, self.model_state):
            bk, wd = list(temp_arc_p)
            model = self.build_single(wd, bk)
            try:
                model.fe.load_state_dict(temp_model_state, strict=False)
                model.mv.weight.data = temp_model_state['out.weight']
                model.mv.bias.data = temp_model_state['out.bias']
            except:
                model.load_state_dict(temp_model_state)

            if freezeFe:
                self.freezon_layers(model.fe.conv_block)

            self.DenseSME.append(model)

    def _modify_keys(self, state):
        if list(state.keys())[0][0:2] == '0.':
            temp_dic_keys = list(state.keys())
            for key in temp_dic_keys:
                state[key[2:]] = state[key]
                del state[key]

    def extract_sort_keys(self, path):
        parts = path.split('\\')[-1].split('_')  # Extract the relevant part of the path
        model_number = int(parts[1])  # Convert model number to integer
        model_size = int(parts[2].split('.')[0])  # Convert model size to integer
        return model_number, model_size

    def narrow_net(self, width=32,flag=True):
        temp = np.asarray(self.arc_p)
        if flag:
            index = temp[:, 1] < width
        else:
            index = temp[:, 1] >= width
        self.arc_p = [item for item, is_selected in zip(self.arc_p, index) if is_selected]
        self.DenseSME = [item for item, is_selected in zip(self.DenseSME, index) if is_selected]
        self.m_name = [item for item, is_selected in zip(self.m_name, index) if is_selected]

    def shallow_net(self, depth=5):
        temp = np.asarray(self.arc_p)
        index = temp[:, 0] < depth
        self.arc_p = [item for item, is_selected in zip(self.arc_p, index) if is_selected]
        self.DenseSME = [item for item, is_selected in zip(self.DenseSME, index) if is_selected]
        self.m_name = [item for item, is_selected in zip(self.m_name, index) if is_selected]

    def freezon_bn_all(self, model):
        for xx, yy in zip(model.named_parameters(), model.parameters()):
            if '.bn' in xx[0]:
                yy.requires_grad = False

    def freezon_bn(self, bn_layer):
        bn_layer.weight.requires_grad = False
        bn_layer.bias.requires_grad = False


    def freezon_layers(self, module):
        for param in module.parameters():
            param.requires_grad = False


def ensemble_results(path2_ftmodels, val_data):
    if isinstance(path2_ftmodels, str):
        modelX = Build_DenseSM(path2_ftmodels)
        modelX.rebuild_DenseSME()
    else:
        modelX = path2_ftmodels
    xtest, ytest = val_data
    all_predicted = []
    model_sta = []
    for i in range(len(modelX.DenseSME)):  #
        with torch.no_grad():
            modelX.DenseSME[i].to(device)
            modelX.DenseSME[i].eval()
            y_pred = modelX.DenseSME[i](xtest)
            if isinstance(y_pred, tuple):
                res = ResultsAnalysis(ytest, y_pred[0])
                all_predicted.append(y_pred[0].to('cpu').numpy())
            else:
                res = ResultsAnalysis(ytest, y_pred)
                all_predicted.append(y_pred.to('cpu').numpy())
        model_sta.append(res.stat)

        # res.scatter_density_fig()

    ensemble_predicted = np.nanmean(np.squeeze(np.asarray(all_predicted)), axis=0)
    all_predicted.append(np.reshape(ensemble_predicted, [ensemble_predicted.shape[0], 1]))
    m_specific_y = np.squeeze(np.asarray(all_predicted))
    res = ResultsAnalysis(ytest, ensemble_predicted)
    res.scatter_density_fig()
    model_sta.append(res.stat)
    df = pd.DataFrame(np.asarray(model_sta), columns=['Bias', 'R', 'RMSE', 'ubRMSE', 'invRate'])
    modelX.m_name.append('ensemble')
    df.index = modelX.m_name
    return df, res, m_specific_y


def CORAL(source, target, **kwargs):
    d = source.data.shape[1]
    ns, nt = source.data.shape[0], target.data.shape[0]
    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm / (ns - 1)

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt / (nt - 1)

    # frobenius norm between source and target
    loss = torch.mul((xc - xct), (xc - xct))
    loss = torch.sum(loss) / (4*d*d)
    return loss


class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss