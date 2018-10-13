#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from data import *
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
# ABS my
def ABS(X):
    return scipy.sqrt(scipy.real(X) ** 2 + scipy.imag(X) ** 2)
def G(x,tao_0):
    r_0=1./tao_0
    return 1. * scipy.exp(-(x/ (r_0)) ** 2)
#DF_2D########################
def massiv2D(f,xx,tt,N,M):
    A = [[0] * M for l in range(N)]
    for i in range(0, N):
        for j in range (M):
            A[i][j] = f(xx[i],tt[j])
    return A

def reverse (A):
    L = np.zeros((np.shape(A)[0], np.shape(A)[1]), dtype=complex)
    for i in range(0, np.shape(A)[0]):
        for j in range(0, np.shape(A)[1]):
            L[np.shape(A)[0]-1-i][j] = (A[i][j])
    return L
def reverse2 (A):
    L = np.zeros((np.shape(A)[0], np.shape(A)[1]), dtype=complex)
    for i in range(0, np.shape(A)[0]):
        for j in range(0, np.shape(A)[1]):
            L[i][np.shape(A)[1]-1-j] = (A[i][j])
    return L
def gr(A, xx, col='b', xname='', yname='', name='',lww=6):

    pylab.plot(xx, A, col,lw=lww)
    pylab.grid(True)

    pylab.ylabel(yname, family="normal", fontsize='44',labelpad=10)
    pylab.xlabel(xname, family="normal", fontsize='44',labelpad=-9)
    #pylab.title(name, family="verdana", fontsize='16')

def gr_2D(A,a,b,c,d, xname='', yname='', name=''):
    max = abs(ABS(A)).max()
    fig=pylab.figure()
    ax=fig.add_axes([0.14,0.15,0.88,0.80])
    im = pylab.imshow(A, interpolation='bilinear', cmap='jet',
                      origin='lower',
                      extent=[a, b, c,
                              d], aspect='auto', vmax=max, vmin=0)

    cb=pylab.colorbar(im,shrink=1,pad=0.02)
    pylab.grid(True)
    xax=ax.get_xaxis()
    yax=ax.get_yaxis()
    #cb.set_ticks([0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000])
    #cb.set_ticks([round(0), round(max / 2, 1)])
    """
    rrr=round(int(round( max,2)),1)

    if rrr==0:
        rrr=max*1.2
        #cb.set_ticks([round(0, 1), round(rrr / 2, 1)])
        
      else:
        cb.set_ticks([round(0,1), round(rrr / 2, 1), round(rrr, 1)])
    if rrr>=100:
        cb.set_ticks([round(0), round(rrr / 2), round(rrr)])

    cb.set_ticks([round(0),round( max/2,1),(round( max,2))])
    print [round(0,1), round(rrr / 2, 1), round(rrr,1)]
    """


    cb.ax.tick_params(labelsize=24)
    xAX=np.linspace(round(a*0.9,1),round(b*0.9,1),3)
    #yAX = np.linspace(round(c, 2), round(d*0.9, 2), 2)
    #yAX =[round(c, 2), round(d*0.8, 2)]
    xlabels=xax.get_ticklabels()
    ylabels = yax.get_ticklabels()
    ax.set_xticks(xAX)
    for label in xlabels:
        label.set_fontsize (24)
    #ax.set_yticks(yAX)
    for label in ylabels:
        label.set_fontsize(24)
    pylab.ylabel(yname, family="verdana", fontsize='26')
    pylab.xlabel(xname, family="verdana", fontsize='26')
    #pylab.title(name, family="verdana", fontsize='26')