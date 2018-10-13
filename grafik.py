#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 19:48:28 2017

@author: xdshiro
"""
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from data import *

l1=z1-(z2-z1)/MM*MMVH
l2=z2*10**3
def grr(A, a, b, c, d, xname='', yname='', name=''):
    max = abs(ABS(A)).max()
    fig = pylab.figure()

    ax = fig.add_axes([0.09, 0.15, 0.90, 0.83])

    im = pylab.imshow(A, interpolation='bilinear', cmap='jet',
                      origin='lower',
                      extent=[a, b, c,
                              d], aspect='auto',norm=LogNorm())

    #,norm=LogNorm()
    #границы ФК
    pylab.plot([a, b], [0, 0], color='yellow', linewidth=4.0)
    pylab.plot([a, b], [l2, l2], color='yellow', linewidth=4.0)
    pylab.xlim(a, b)
    pylab.ylim(zVIH,zVH)
    #formatter = LogFormatter(10)
    #cb = pylab.colorbar(im, shrink=1, pad=0.02)
    pylab.grid(True)
    xax = ax.get_xaxis()
    yax = ax.get_yaxis()
    print max
    #cb.set_ticks([0,2])
    #cb.set_ticks([0,0.8,0.4])
    cb = pylab.colorbar(im, shrink=1, pad=0.02)
    cb.set_ticks([0.1,1,10,100,1000,10000,100000,1000000,10000000,100000000])
    #cb.set_ticks([round(0), round(max / 2, 1)])
    #ax.text(0.75, 0.4, '(b)', color='white', family="verdana", fontsize='28')


    xAX = np.linspace(round(a * 0.9, 1), round(b * 0.9, 1), 3)

    yAX = np.linspace(z1*10**3, z2*10**3, 2)

    xlabels = xax.get_ticklabels()
    ylabels = yax.get_ticklabels()
    ax.set_xticks(xAX)
    ax.set_yticks(yAX)
    for label in xlabels:
        label.set_fontsize(28)
    for label in ylabels:
        label.set_fontsize(28)
    pylab.ylim(zVIH*10**3, zVH*10**3)
    pylab.ylabel(yname, family="verdana", fontsize='28')
    ax.yaxis.set_label_coords(-0.04, 0.5)
    pylab.xlabel(xname, family="verdana", fontsize='28')
    ax.xaxis.set_label_coords(0.5, -0.07)

    """
    cb = pylab.colorbar(im, ax=ax, orientation='horizontal', shrink=0.83, pad=-0.065)
    if max <= 1:
        cb.set_ticks([round(0, 0), 0.5])
    else:
        rrr = max * 1.2
        cb.set_ticks([round(0, 0), round(rrr / 2, 0), round(rrr, 0)])
    """


    fg_color = 'white'
    cb.ax.xaxis.set_ticks_position('top')
    # cb.set_label('colorbar label', color='white')
    cb.ax.tick_params(labelsize=14, which='major', color='black')
    # set colorbar edgecolor
    cb.outline.set_color('black')
    pylab.setp(pylab.getp(cb.ax.axes, 'xticklabels'), color='black')
    cb.ax.tick_params(labelsize=28)
    #pylab.tight_layout(h_pad=-0.1, w_pad=0.15)

    arrowprops = {
        'arrowstyle': 'simple',
        'linewidth': 5,
        'color': 'yellow',
    }

    if tt==0:
        ax.annotate(u'',
                    xy=(-0.03, 0.06),
                    xytext=(0.04, -0.06),
                    arrowprops=arrowprops)
    else:
        ax.annotate(u'',
                    xy=(0.03, 0.06),
                    xytext=(-0.04, -0.06),
                    arrowprops=arrowprops)

    # pylab.title(name, family="verdana", fontsize='26')

from data import *
zVH=z1-(z2-z1)/MM*MMVH
zVIH = z2 + (z2 - z1) / MM * MMVIH
fileNAME='new_tetta=-29.5, g*=0.005'
tt=0
fl_open=open(fileNAME,'r')

ZZZ=np.loadtxt(fl_open,dtype=complex)
ZZZ=ABS(ZZZ+0.01)
fl_open.close()
#рисуем

grr(ABS(ZZZ), a * 1000, b * 1000, zVIH * 1000, zVH * 1000, xname=u'x(mm)', yname=u'z(mm)', name=namef)



pylab.savefig(fileNAME + "log.png", dpi=60)
pylab.show()