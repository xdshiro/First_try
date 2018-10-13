#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 19:48:28 2017

@author: xdshiro
"""
import math
import scipy
import pylab
import numpy as np
import scipy.integrate as integrate
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift, fftfreq
from funcs import *

# parameters
C = 299792458.

if 1 == 1:
    N, M = 140, 140
    MMVH = 50
    MM = 180
    MMVIH = 50
    #MMVH =45
    #MM = 115
    #MMVIH = 45
    # 180
    v_x=0
    # M/2=0

    r_0 = 6000 * 10 ** (-6)
    tao_0 = 1. * (10 ** (-12))

    timeS = [-1 * tao_0,3 * tao_0,7 * tao_0,11 * tao_0,15 * tao_0]
    #timeS =  [15*tao_0]
    #z1, z2 = round(C*tao_0,3)*(0), round(C*tao_0,4)*1.9
    #z1, z2 = 0.0 * 10 ** (-3), 2300 * 10 ** (-3)
    z1, z2 = 0.0 * 10 ** (-3), 1.8 * 10 ** (-3)

    #Chirp
    # НЕ ЗАБЫТЬ ПРОВЕРИТЬ ПОЛОЖЕНИЕ СИГМЫ И БЕТТЫ
    Sigma=0
    #Betta_S=[-10,-15,-20,-25,-30,-35,-40,-50,-45,-55,-57.5,-60]
    Betta_S=[0]
    Betta=0

    tetta = scipy.pi / 180 * (-30)

    tettaS = [scipy.pi / 180 * (30),scipy.pi / 180 * (-30)]
    #tettaS = -1
    at = 15

    #a, b = -round(C*tao_0,4)*1.2 ,round(C*tao_0,4)*1.2
    #a, b = -1000. * 10 ** (-3), 1000. * 10 ** (-3)
    a, b = -1. * 10 ** (-3), 1. * 10 ** (-3)
    c, d = -1. * at * tao_0, at * tao_0

    D = 1.0 * 10 ** (-6)
    # Hi_0 = 2.25
    Hi_0 = 2.25
    lam = 1.0 * 10 ** (-6)
    # T2 odn
    gamma_0 = 0.005
    # T2* neod
    gamma_1 = 0.2
    gamma_1S=[0.005,0.1]
    epsi = 0.9999

# имя сохраненного файла
if 1 == 1:
    # no time

    fileNAME2 = u'tetta='+ str(
        (tetta * 180 / scipy.pi)) + u', g*=' + str(gamma_1)
    namef='1ns'
    fileNAME2='smotrim4'
    """
    fileNAME1  = fileNAME2
    nameLIN = u'g=' + str(gamma_1) + u'_tao=' + str(tao_0 * 10 ** 12)
    """


w_0 = 2. * scipy.pi * C / lam
k_0 = w_0 / C
h = 2. * scipy.pi / D

# superfast
superFAST = 2
# obshee sohranenie
save1 = 2
save2 = 1
# norm=1 - dispersion, norm=2 - without (classic)
norm = 1
# поле 0-прямая, h-дифрагированная
pole = 'kl'
# eqw. g: 1 - |-|, 2 - Gauss, 3 - Lorenz
prop = 1
I = 5000
# st=2 - intensivnost' ( это степень)
st = 2

lin = 2
alfaF = 1
# Amp in eqw. g !!!
# для 0.005 0.05 (но не очень ровно) 0.0952
# alfa_00 = 0.25585

#alfa_00 = 5.55717232187 * 10 ** (-17)
alfa_00 = 5.4081714279e-18
# both classic and dissp.
a_k = 0.008

b_k = 0.0079999

# гаммы

# интервалы
if 1 == 1:
    a1, b1 = -1. * 2 * scipy.pi * N / (2 * (b - a)), 1. * 2 * scipy.pi * N / (2 * (b - a))
    c1, d1 = -1. * 2 * scipy.pi * M / (2 * (d - c)), 1. * 2 * scipy.pi * M / (2 * (d - c))
    xx = np.linspace(a, b, N)
    tt = np.linspace(c, d, M)
    kk = np.linspace(a1, b1, N)
    ww = np.linspace(c1, d1, M)
    #################################