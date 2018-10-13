#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 19:48:28 2017

@author: xdshiro
"""
# Интегрирование, массивы
from data import *

# odnorodnoe ushirenie

for gamma_1 in gamma_1S:
    if 1 == 1:
        gamma_g = gamma_1 * w_0
        gamma_a = gamma_0 * w_0
        betta_0 = alfa_00 * gamma_0 * w_0 ** 2
        delta = 2 * gamma_g
    def e_a(w, w1_0):
        return -1 * betta_0 / (-1. * w1_0 + w + 1j * gamma_a)


    def timeFIND(t, tt, M):
        flag = 10
        iflag = 0
        t = 1. * t
        for i in range(0, M):
            if (ABS((tt[i]) + (t)) < flag):
                flag = ABS(tt[i] + t)
                iflag = i
        print "Точное время: ", tt[iflag], " (", iflag, ")"
        return iflag


    # forma neodnorodnogo ushireniya
    if prop == 3:
        print 'Lorenz'


        def g(t):
            return gamma_g / ((t ** 2. + gamma_g ** 2.) * scipy.pi)
    else:
        if prop == 2:
            print 'Gauss'


            def g(t):
                return (1. / (gamma_g * (scipy.pi ** 0.5))) * scipy.exp(-t ** 2. / (gamma_g ** 2.))
        else:
            print '|-|'


            def g(t):
                if ABS(t) <= (delta / 2.):
                    return 1. / delta
                else:
                    return 0.


    # temp
    def f_e_rez(w1_0):
        return e_a(w, w1_0) * g(w1_0 - w_0)


    # IMPULSE shape
    def A_in_t_bezchirp(x, t):
        return 1. * scipy.exp(
            -(x * scipy.cos(tetta) / r_0) ** 2 - ((1. / tao_0) ** 2) * (t + x * scipy.sin(tetta) / C) ** 2)

    #IMPULSE chirp
    def A_in_t(x, t):
        return 1. * scipy.exp(
            -(x * scipy.cos(tetta) / r_0) ** 2*(1-1j*Sigma) - ((1.-1j*Betta) / (tao_0) ** 2) * (t + x * scipy.sin(tetta) / C) ** 2)
    pad=np.zeros((N, M), dtype=complex)
    for i in range (N):
        for j in range (M):
            pad[i,j]=A_in_t_bezchirp(xx[i],tt[j])
    """
    print pad
    gr_2D(ABS(pad),a,b,c,d)
    pylab.show()
    """
    def disp():
        ylist = np.zeros((M), dtype=complex)
        ylist2 = np.zeros((I), dtype=complex)
        ylistGAUSS = np.zeros((M), dtype=complex)
        E_i = np.zeros((M), dtype=complex)

        print gamma_1
        ii = 0
        for w in np.linspace(w_0 + c1, w_0 + d1, M):
            iii = 0
            for w1_0 in np.linspace(w - gamma_1 * w_0 * 3, w + gamma_1 * w_0 * 3, I):
                ylist2[iii] = e_a(w, w1_0) * g(w1_0 - w_0)
                iii = iii + 1
            ylist[ii] = integrate.simps(ylist2, np.linspace(w - gamma_1 * w_0 * 3, w + gamma_1 * w_0 * 3, I))
            # ylist3[ii] = scipy.exp(-((w_0 - w) * tao_0) ** 2) / 100
            global E_i
            E_i[ii] = ylist[ii]
            ylist[ii] = g(w - w_0)
            ylistGAUSS[ii] = G(w - w_0, tao_0 * scipy.pi)
            ii = ii + 1
        # liniya

        if lin == 1:
            xlist = np.linspace((w_0 + c1) / w_0, (w_0 + d1) / w_0, M)
            maxx = abs(ABS(scipy.imag(E_i))).max()
            fig = pylab.figure()
            ax = fig.add_axes([0.14, 0.15, 0.80, 0.80])
            gr(scipy.imag(E_i) / maxx, xlist, 'b')
            gr(scipy.real(E_i) / maxx, xlist, 'r')
            maxx = abs(ABS(ylist)).max()
            # gr(ylist/maxx*0.5, xlist, 'g')
            maxx = abs(ABS(ylistGAUSS)).max()
            gr(ylistGAUSS, xlist, 'g', xname=u'\u03C9/\u03C9$_0$ (a.u.)', yname=u'amplitude (a.u.)')

            xax = ax.get_xaxis()
            yax = ax.get_yaxis()
            xAX = [round(0.9995, 4), round(1.0000, 1), round(1.0005, 4)]
            yAX = np.linspace(0, 1, 3)
            xlabels = xax.get_ticklabels()
            ylabels = yax.get_ticklabels()
            ax.set_xticks([round(0.9994, 4), round(1.0000, 1), round(1.0006, 4)], True)
            for label in xlabels:
                label.set_fontsize(20)
            ax.set_yticks(yAX)
            for label in ylabels:
                label.set_fontsize(20)

            if save == 1:
                pylab.savefig(nameLIN + ".jpg", dpi=200)
            pylab.show()

        maxx = abs(ABS(scipy.imag(E_i))).max()
        print maxx
        return maxx

    alfa_00=alfa_00
    #здесь и появляется disp()
    def alfaFIND():
        b_kk=epsi*a_k
        asdf = (alfa_00 / ((disp()) / b_kk))
        print asdf
        return asdf
    """
    for tetta in tettaS:
        for gamma_1 in gamma_1S:
    """




    def w(W):
        return w_0 + W


    def k(W):
        return w(W) / C


    # i1
    def integ1(N, a, b, f):
        x = np.linspace(a, b, N)
        Y = [0] * N
        for i in range(0, N):
            Y[i] = f(x[i])
        return integrate.simps(Y, x)



    xx = np.linspace(a, b, N)
    tt = np.linspace(c, d, M)
    kk = np.linspace(a1, b1, N)
    ww = np.linspace(c1, d1, M)
    ZZ_0 = np.zeros((N, M), dtype=complex)
    ZZ_h = np.zeros((N, M), dtype=complex)
    ZZ_0VIH = np.zeros((N, M), dtype=complex)
    ZZ_hVIH = np.zeros((N, M), dtype=complex)
    ZZ_0_i = np.zeros((N, M), dtype=complex)
    ZZ_h_i = np.zeros((N, M), dtype=complex)
    ZZZ = np.zeros((MM, N), dtype=complex)
    field = np.zeros((N, MM), dtype=complex)
    vremya = np.zeros(M, dtype=complex)# это в определенной точке во времени
    #vremya = np.zeros((MM), dtype=complex)
    R_sMAS = np.zeros((N, M), dtype=complex)
    # vihod
    # спектр
    print gamma_1


    """
    fileNAME2 = u'9.9' + u', g=' + str(gamma_1) + u', S=' + str(Sigma) + u', B=' + str(Betta) + u', tao=' + str(
        tao_0 * 10 ** 12) + u'пс, tetta=' + str(
        (tetta * 180 / scipy.pi))
    fileNAME2 = u'abs_tao=' + str(tao_0 * 10 ** 12) + u'пс, time=' + u'пс, ' + str(
        (tetta * 180 / scipy.pi)) + u', Sigma=' + str(Betta)
    """
    if alfaF == 1 and norm==1:
        alfa_00=alfaFIND()
        betta_0 = alfa_00 * gamma_0 * w_0 ** 2

    if norm==1:
        disp ()
    for tetta in tettaS:
        fileNAME2 =namef +  u', tetta=' + str(
            (tetta * 180 / scipy.pi)) + u', g*=' + str(gamma_1)
        Z = fftshift(fft2(massiv2D(A_in_t, xx, tt, N, M)))

        #full field
        fieldFULL = np.zeros((MM+MMVIH+MMVH,N), dtype=complex)
        """
        if superFAST == 1:
            Z_fast = np.zeros((N, M), dtype=complex)
            Z_fastTR = np.zeros((M,  N), dtype=complex)
            if tetta > 0:
                k_0x = ABS(k_0 * scipy.sin(tetta))
            else:
                h = -2. * scipy.pi / D
                k_0x = -1.0 * ABS(k_0 * scipy.sin(tetta))
            for z in zFAST:
                print z
                for i in range(0, N):
                    for j in range(0, M):
                        KKK, WWW = kk[i], ww[j]
                        if norm == 1:
                            b_k = -1. * 1j * E_i[j]
                        if tetta > 0:
                            Hi_h = 0.5 * (a_k - b_k)
                            Hi__h = 0.5 * (a_k + b_k)
                        else:
                            Hi_h = 0.5 * (a_k + b_k)
                            Hi__h = 0.5 * (a_k - b_k)
                        q_0x = k_0x + KKK
                        k_z = scipy.sqrt((k(WWW)) ** 2 - q_0x ** 2)
                        alfa_0 = q_0x - h / 2
                        R_1 = (alfa_0 * h - (alfa_0 ** 2 * h ** 2 + Hi_h * Hi__h * k(WWW) ** 4) ** 0.5) / (
                            Hi__h * k(WWW) ** 2)
                        R_2 = (alfa_0 * h + (alfa_0 ** 2 * h ** 2 + Hi_h * Hi__h * k(WWW) ** 4) ** 0.5) / (
                            Hi__h * k(WWW) ** 2)
                        WW = (alfa_0 ** 2 * h ** 2 + Hi_h * Hi__h * k(WWW) ** 4) ** 0.5
                        q_0z1 = (Hi_0 * k(WWW) ** 2 - q_0x ** 2 + alfa_0 * h - WW) ** 0.5
                        q_0z2 = (Hi_0 * k(WWW) ** 2 - q_0x ** 2 + alfa_0 * h + WW) ** 0.5
                        R_12 = R_1 - R_2
                        f_s = (q_0z2 * R_1 - q_0z1 * R_2) / R_12
                        R_s = (k_z - f_s) / (k_z + f_s)
                        # Отражение есть-1, нет-0
                        otr = 0
                        global A_01
                        A_01 = (1. + otr * R_s) * (
                            alfa_0 * h + (alfa_0 ** 2 * h ** 2 + Hi_h * Hi__h * k(WWW) ** 4) ** 0.5) / (
                                   2 * (alfa_0 ** 2 * h ** 2 + Hi_h * Hi__h * k(WWW) ** 4) ** 0.5)
                        global A_h1
                        A_h1 = R_1 * A_01
                        global A_02
                        A_02 = (1. + otr * R_s) * (
                            -alfa_0 * h + (alfa_0 ** 2 * h ** 2 + Hi_h * Hi__h * k(WWW) ** 4) ** 0.5) / (
                                   2 * (alfa_0 ** 2 * h ** 2 + Hi_h * Hi__h * k(WWW) ** 4) ** 0.5)
                        global A_h2
                        A_h2 = R_2 * A_02
                        if pole == 'O':
                            A_h1 = 0
                            A_h2 = 0
                        if pole == 'h':
                            A_01 = 0
                            A_02 = 0
                        Z_fast[i][j] = ZZ_0[i][j] + ZZ_h[i][j]
            # gr_2D(ABS(Z_fast),a,b,c,d)
            # pylab.show()
            Z_fast = ABS(ifft2(Z_fast))
        
            for i in range(0, N):
                for j in range(0, M):
                    Z_fastTR[j][i] = (ABS(Z_fast[i][j])) ** (st)
        
            gr_2D(Z_fastTR, a * 1000, b * 1000, c / tao_0, d / tao_0, xname=u'x(mm)', yname=u'z(mm)', name=name)
            pylab.ylim(-1 * at, 0)
            pylab.show()
        
        
        """
        timeI=-1
        for time in timeS:
            timeI=timeI+1
            print time
            time = timeFIND(time, tt, M)
            print time
            if tetta > 0:
                k_0x = ABS(k_0 * scipy.sin(tetta))
            else:
                h = -2. * scipy.pi / D
                k_0x = -1.0 * ABS(k_0 * scipy.sin(tetta))
            zzz = 0
            for z in np.linspace(z1, z2, MM):
                print zzz
                for i in range(0, N):
                    for j in range(0, M):
                        KKK, WWW = kk[i], ww[j]
                        if norm == 1:
                            b_k = -1. * 1j * E_i[j]
                        if tetta > 0:
                            Hi_h = 0.5 * (a_k - b_k)
                            Hi__h = 0.5 * (a_k + b_k)
                        else:
                            Hi_h = 0.5 * (a_k + b_k)
                            Hi__h = 0.5 * (a_k - b_k)
                        q_0x = k_0x + KKK
                        k_z = scipy.sqrt((k(WWW)) ** 2 - q_0x ** 2)
                        alfa_0 = q_0x - h / 2
                        R_1 = (alfa_0 * h - (alfa_0 ** 2 * h ** 2 + Hi_h * Hi__h * k(WWW) ** 4) ** 0.5) / (
                            Hi__h * k(WWW) ** 2)
                        R_2 = (alfa_0 * h + (alfa_0 ** 2 * h ** 2 + Hi_h * Hi__h * k(WWW) ** 4) ** 0.5) / (
                            Hi__h * k(WWW) ** 2)
                        WW = (alfa_0 ** 2 * h ** 2 + Hi_h * Hi__h * k(WWW) ** 4) ** 0.5
                        q_0z1 = (Hi_0 * k(WWW) ** 2 - q_0x ** 2 + alfa_0 * h - WW) ** 0.5
                        q_0z2 = (Hi_0 * k(WWW) ** 2 - q_0x ** 2 + alfa_0 * h + WW) ** 0.5

                        R_12 = R_1 - R_2
                        f_s = (q_0z2 * R_1 - q_0z1 * R_2) / R_12
                        R_s = (k_z - f_s) / (k_z + f_s)

                        # Отражение есть-1, нет-0
                        otr = 1
                        A_01 = (1. + otr * R_s) * (
                            alfa_0 * h + (alfa_0 ** 2 * h ** 2 + Hi_h * Hi__h * k(WWW) ** 4) ** 0.5) / (
                                   2 * (alfa_0 ** 2 * h ** 2 + Hi_h * Hi__h * k(WWW) ** 4) ** 0.5)

                        A_h1 = R_1 * A_01

                        A_02 = (1. + otr * R_s) * (
                            -alfa_0 * h + (alfa_0 ** 2 * h ** 2 + Hi_h * Hi__h * k(WWW) ** 4) ** 0.5) / (
                                   2 * (alfa_0 ** 2 * h ** 2 + Hi_h * Hi__h * k(WWW) ** 4) ** 0.5)

                        A_h2 = R_2 * A_02
                        if pole == '0':
                            A_h1 = 0
                            A_h2 = 0
                        if pole == 'h':
                            A_01 = 0
                            A_02 = 0

                        ZZ_0[i][j] = (Z[i][j]) * (A_01 * scipy.exp(1j * q_0z1 * z) + A_02 * scipy.exp(1j * q_0z2 * z))
                        ZZ_h[i][j] = (Z[i][j]) * (A_h1 * scipy.exp(1j * q_0z1 * z) + A_h2 * scipy.exp(1j * q_0z2 * z))
                        if zzz == 0:
                            R_sMAS[i][j]= R_s
                        if zzz == MM-1:
                            ZZ_0VIH[i][j] = (Z[i][j]) * (A_01 * scipy.exp(1j * q_0z1 * z) + A_02 * scipy.exp(1j * q_0z2 * z))
                            ZZ_hVIH[i][j] = (Z[i][j]) * (A_h1 * scipy.exp(1j * q_0z1 * z) + A_h2 * scipy.exp(1j * q_0z2 * z))
                ZZ_h_i = ifft2(ZZ_h)
                ZZ_0_i = ifft2(ZZ_0)
                for nn in range(N):
                    field[nn][zzz] =  ((ZZ_0_i[nn][time]))+((ZZ_h_i[nn][time]*scipy.exp(-1j*h*xx[nn])))

                #1ое - икс на определенном игрик и т
                if 1==2:

                    vremya = (ABS(ZZ_0_i[v_x,0:M/2] + ZZ_h_i[v_x,0:M/2] * scipy.exp(-1j * h * xx[v_x]))**(st))
                    vrr= (ABS(ZZ_0_i[v_x,0:M/2] + ZZ_h_i[v_x,0:M/2] * scipy.exp(-1j * h * xx[v_x]))**(st))
                    #vremya = ABS(ZZ_0_i[v_x, :] + ZZ_h_i[v_x, :] * scipy.exp(-1j * h * xx[v_x])) ** (st)
                    print 'координата икс '
                    print xx[v_x]
                    print z

                    for fff in range(0,M/2):

                        vremya[fff]=vrr[M/2-1-fff]

                    #gr(vremya, tt[:])
                    fig = pylab.figure()

                    ax = fig.add_axes([0.14, 0.15, 0.80, 0.80])
                    gr(vremya,tt[(M/2):M]*10**12, xname=u't(ps)', yname='amplitude(a.u.)', col='b')

                    xax = ax.get_xaxis()
                    yax = ax.get_yaxis()
                    xAX = [0, 5, 10]
                    #pylab.xlim(-0.8, 0.8)
                    pylab.xlim(0, 10)

                    #yAX = [0,0.5,1]
                    xlabels = xax.get_ticklabels()
                    ylabels = yax.get_ticklabels()
                    ax.set_xticks(xAX)
                    for label in xlabels:
                        label.set_fontsize(20)
                    #ax.set_yticks(yAX)
                    for label in ylabels:
                        label.set_fontsize(20)

                    #pylab.savefig("n+0495.jpg", dpi=200)

                    pylab.show()
                #2ое время на икс и з
                if 1 == 2:

                    vremya = (ABS(ZZ_0_i[v_x, :] + ZZ_h_i[v_x, :] * scipy.exp(-1j * h * xx[v_x])) ** (st))
                    vrr = (ABS(ZZ_0_i[v_x, :] + ZZ_h_i[v_x, :] * scipy.exp(-1j * h * xx[v_x])) ** (st))
                    # vremya = ABS(ZZ_0_i[v_x, :] + ZZ_h_i[v_x, :] * scipy.exp(-1j * h * xx[v_x])) ** (st)
                    print 'координата икс '
                    print xx[v_x]
                    print z

                    for fff in range(0,M):
                        vremya[fff] = vrr[M - 1 - fff]

                    # gr(vremya, tt[:])
                    fig = pylab.figure()

                    ax = fig.add_axes([0.14, 0.15, 0.80, 0.80])
                    gr(vremya, tt * 10 ** 9, xname=u't(ns)', yname='amplitude(a.u.)', col='b')

                    xax = ax.get_xaxis()
                    yax = ax.get_yaxis()
                    xAX = [-4,-2,0,2,4]
                    # pylab.xlim(-0.8, 0.8)
                    pylab.xlim(-5, 5)

                    # yAX = [0,0.5,1]
                    xlabels = xax.get_ticklabels()
                    ylabels = yax.get_ticklabels()
                    ax.set_xticks(xAX)
                    for label in xlabels:
                        label.set_fontsize(20)
                    # ax.set_yticks(yAX)
                    for label in ylabels:
                        label.set_fontsize(20)

                    pylab.savefig("1000_05+0.jpg", dpi=200)

                    pylab.show()
                zzz = zzz + 1
            for i in range(0, N):
                for j in range(0, MM):
                    ZZZ[MM - 1 - j][i] = ABS((field[i][j]) ** (st))

            # рисуем
            """
            gr_2D((ZZZ), a * 1000, b * 1000, z2 * 1000, z1 * 1000, xname=u'x(mm)', yname=u'z(mm)', name=namef)
            if save == 1:
                pylab.savefig(nameS + ".jpg", dpi=200)
            pylab.show()
            """



            fieldVH = np.zeros((N, MMVH), dtype=complex)

            zVH=z1-(z2-z1)/MM*MMVH
            print zVH
            zzz=0
            for z in np.linspace(0, zVH, MMVH):

                for i in range(0, N):
                    for j in range (0,M):
                        KKK, WWW = kk[i], ww[j]
                        q_0x = k_0x + KKK
                        k_z = scipy.sqrt((k(WWW)) ** 2 - q_0x ** 2)
                        ZZ_0[i][j] = R_sMAS[i][j]*Z[i][j] * scipy.exp(1 * 1j * k_z * (z))+Z[i][j]* scipy.exp(1 * -1j * k_z * (z))


                ZZ_0_i = reverse2(ifft2(ZZ_0))
                for nn in range(N):
                    fieldVH[nn][zzz] =  ZZ_0_i[nn][time]
                zzz = zzz + 1
            fieldVH=ABS(reverse2(ABS(fieldVH.transpose()**(st))))
            # отдельно выход

            """
            gr_2D(fieldVH, a * 1000, b * 1000, zVH * 1000, z1 * 1000, xname=u'x(mm)', yname=u'z(mm)', name=namef)
            pylab.show()
            """

            SS=np.vstack((ZZZ,fieldVH))

            #вход+кристалл
            """
            gr_2D(SS, a * 1000, b * 1000, z2 * 1000, zVH * 1000, xname=u'x(mm)', yname=u'z(mm)', name=namef)
            pylab.show()
            """


            fieldVIH = np.zeros((N, MMVIH), dtype=complex)
            zVIH=z2+(z2-z1)/MM*MMVIH
            print zVIH
            zzz=0
            for z in np.linspace(z2, zVIH, MMVIH):


                for i in range(0, N):
                    for j in range (0,M):
                        KKK, WWW = kk[i], ww[j]
                        q_0x = k_0x + KKK
                        k_z = scipy.sqrt((k(WWW)) ** 2 - q_0x ** 2)
                        ZZ_0[i][j] = ZZ_0VIH[i][j] * scipy.exp(1 * 1j * k_z * (z - z2))
                        ZZ_h[i][j] = ZZ_hVIH[i][j] * scipy.exp(1 * 1j * k_z * (z - z2))

                ZZ_h_i = (ifft2((ZZ_h)))
                ZZ_h_i = reverse(ZZ_h_i)
                ZZ_0_i = ifft2(ZZ_0)

                for nn in range(N):
                    fieldVIH[nn][zzz] = ZZ_h_i[nn][time] + ZZ_0_i[nn][time]

                zzz = zzz + 1

            fieldVIH=ABS(reverse(ABS(fieldVIH.transpose()**(st))))
            S=np.vstack((fieldVIH,SS))
            #gr_2D(ABS(S), a * 1000, b * 1000, zVIH * 1000, zVH * 1000, xname=u'x(mm)', yname=u'z(mm)', name=namef)
            #pylab.show()
            if save1==1:
                fileNAME1 = u'tao='+str(tao_0*10**12)+u'пс, time='+str(timeS[timeI]*10**12)+u'пс, '+str((tetta*180/scipy.pi))+u', g*=' + str(gamma_1)
                fl_open = open(fileNAME1, 'w')
                np.savetxt(fl_open, S)
                fl_open.close()
            fieldFULL=fieldFULL+S
        if save2==1:
            fl_open = open(fileNAME2, 'w')
            np.savetxt(fl_open, fieldFULL)
            fl_open.close()

        gr_2D(ABS(fieldFULL), a * 1000, b * 1000, zVIH * 1000, zVH * 1000, xname=u'x(mm)', yname=u'z(mm)', name=namef)


        if save2==1:
            pylab.savefig(fileNAME2 + ".png", dpi=100)
            fl_open.close()
        pylab.close()