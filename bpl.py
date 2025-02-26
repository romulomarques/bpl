import sys
import os
import numpy as np
from numpy.linalg import norm, solve
import pandas as pd
import time
from rmsd import rmsd
from read_sol import read_sol
from create_instances_functions import distance

loopDefs = pd.read_csv('table_loopDefs.csv')

def read_dat(fdat):
    print('Reading ' + fdat)
    df = {'ANCHOR':[], 'CONSTR':[]}
    with open(fdat, 'r') as fid:
        for row in fid:
            if 'ANCHOR' in row:
                row = row.split()
                x = np.array([float(row[-3]), float(row[-2]), float(row[-1])], dtype=float)
                i = int(row[-4]) - 1
                df['ANCHOR'].append({'i': i, 'x': x, 'res': row[-6]})
            elif 'INTERV' in row:
                row = row.split()
                ires = row[-8]
                jres = row[-7]
                iatm = row[-6]
                jatm = row[-5]
                i = int(row[-4]) - 1
                j = int(row[-3]) - 1
                lij = float(row[-2])
                uij = float(row[-1])
                df['INTERV'] = {'i': i, 'j': j, 'lij': lij, 'uij': uij,
                                'ires': ires, 'jres': jres, 'iatm': iatm, 'jatm': jatm}
            elif 'DATA' in row:
                row = row.split()
                ires = row[-8]
                jres = row[-7]
                iatm = row[-6]
                jatm = row[-5]
                i = int(row[-4]) - 1
                j = int(row[-3]) - 1
                lij = float(row[-2])
                uij = float(row[-1])
                df['CONSTR'].append({'i': i, 'j': j, 'lij': lij, 'uij': uij,
                                'ires': ires, 'jres': jres, 'iatm': iatm, 'jatm': jatm})
        df['ANCHOR'] = sorted(df['ANCHOR'], key=lambda u: u['i'])
        df['CONSTR'] = sorted(df['CONSTR'], key=lambda u: (u['i'], u['j']))
    return df 


# Removing redundancy from distance information.
def cleanD(p, D):
    S = {}
    q = np.zeros(len(p), dtype=int)
    for k, i in enumerate(p):
        q[i] = k
        S[i] = {}

    for i in range(len(p)):
        p_i = p[i]
        for j in D[p_i]:
            # q[j] < q
            if q[j] < i:
                S[p_i][j] = D[p_i][j]

    return S


# Removing redundancy from distance information and reorganizing it according to the vertex order 'p'.
def cleanAndReorganizeD(p, D):
    S = {}
    q = np.zeros(len(p), dtype=int)
    for k, i in enumerate(p):
        q[i] = k
        S[i] = {}

    for i in range(len(p)):
        p_i = p[i]
        for j in range(i):
            if p[j] in D[p_i]:
                S[p_i][p[j]] = D[p_i][p[j]]

    return S


# vertex order presented in the "Distance geometry and protein loop modeling" article, published at Jornal of
# Computational Chemistry, Wiley, 2022.
def Labiak_Lavor_Souza_order(df):
    # map atom type to each residue index
    RESIDUES = {}
    for c in df['CONSTR']:
        if c['ires'] not in RESIDUES:
            RESIDUES[c['ires']] = {}
        if c['jres'] not in RESIDUES:
            RESIDUES[c['jres']] = {}
        RESIDUES[c['ires']][c['iatm']] = c['i']
        RESIDUES[c['jres']][c['jatm']] = c['j']
    RESIDUES = {i: RESIDUES[i] for i in sorted(RESIDUES)}
    
    # anchors and first N
    p = [u['i'] for u in df['ANCHOR']]
    p.append(df['INTERV']['i'])

    # append remaining in N, CA, C order
    RESIDUES = [RESIDUES[k] for k in RESIDUES]
    for residue in RESIDUES:
        if residue['N'] not in p:
            p.append(residue['N'])
        if residue['CA'] not in p:
            p.append(residue['CA'])
        if residue['C'] not in p:
            p.append(residue['C'])
    return p


def rigid_body_clockwise_order(RESIDUES, firstres, lastres, endmiddle):
    p = []

    # Adding atoms of the middle of the current rigid body
    for i in range(firstres + 1, endmiddle):
        if RESIDUES[i]['N'] not in p:
            p.append(RESIDUES[i]['N'])
        if RESIDUES[i]['CA'] not in p:
            p.append(RESIDUES[i]['CA'])
        if 'H' in RESIDUES[i]:
            if RESIDUES[i]['H'] not in p:
                p.append(RESIDUES[i]['H'])
        if RESIDUES[i]['C'] not in p:
            p.append(RESIDUES[i]['C'])
        if 'HA' in RESIDUES[i]:
            if RESIDUES[i]['HA'] not in p:
                p.append(RESIDUES[i]['HA'])

    if RESIDUES[lastres]['N'] not in p:
        p.append(RESIDUES[lastres]['N'])
    if RESIDUES[lastres]['CA'] not in p:
        p.append(RESIDUES[lastres]['CA'])
    if 'H' in RESIDUES[lastres]:
        if RESIDUES[lastres]['H'] not in p:
            p.append(RESIDUES[lastres]['H'])

    return p


def rigid_body_anti_clockwise_order(RESIDUES, lastres, beginmiddle):
    p = []

    if lastres + 1 <= beginmiddle:
        if RESIDUES[beginmiddle - 1]['C'] not in p:
            p.append(RESIDUES[beginmiddle - 1]['C'])

        if RESIDUES[beginmiddle - 1]['CA'] not in p:
            p.append(RESIDUES[beginmiddle - 1]['CA'])

            # Note this "H" is from a neighbour residue.
            if beginmiddle != len(RESIDUES):
                if 'H' in RESIDUES[beginmiddle]:
                    if RESIDUES[beginmiddle]['H'] not in p:
                        p.append(RESIDUES[beginmiddle]['H'])

        if beginmiddle - 1 > lastres:
            if RESIDUES[beginmiddle - 1]['N'] not in p:
                p.append(RESIDUES[beginmiddle - 1]['N'])
            if 'HA' in RESIDUES[beginmiddle - 1]:
                if RESIDUES[beginmiddle - 1]['HA'] not in p:
                    p.append(RESIDUES[beginmiddle - 1]['HA'])

        # Adding atoms of the middle of the current rigid body
        for i in sorted(range(lastres + 1, beginmiddle - 1), reverse=True):
            if RESIDUES[i]['C'] not in p:
                p.append(RESIDUES[i]['C'])
            if RESIDUES[i]['CA'] not in p:
                p.append(RESIDUES[i]['CA'])
            # Note this "H" is from a neighbour residue.
            if 'H' in RESIDUES[i + 1]:
                if RESIDUES[i + 1]['H'] not in p:
                    p.append(RESIDUES[i + 1]['H'])
            if RESIDUES[i]['N'] not in p:
                p.append(RESIDUES[i]['N'])
            if 'HA' in RESIDUES[i]:
                if RESIDUES[i]['HA'] not in p:
                    p.append(RESIDUES[i]['HA'])

        # Adding atoms of the first border, which will be the second one in the INVERSE orientation,
        # of the current rigid body
        if RESIDUES[lastres]['C'] not in p:
            p.append(RESIDUES[lastres]['C'])
        if RESIDUES[lastres]['CA'] not in p:
            p.append(RESIDUES[lastres]['CA'])
        # Note this "H" is from a neighbour residue.
        if 'H' in RESIDUES[lastres + 1]:
            if RESIDUES[lastres + 1]['H'] not in p:
                p.append(RESIDUES[lastres + 1]['H'])

    return p


def hydrogen_order(df):
    FIXED_RES = [int(df['ANCHOR'][0]['res']), int(df['ANCHOR'][1]['res']), int(df['ANCHOR'][2]['res'])]

    ires = int(df['INTERV']['ires'])
    jres = int(df['INTERV']['jres'])
    iatm = (df['INTERV']['iatm'])
    jatm = (df['INTERV']['jatm'])

    fixedhares = -1
    closehres = -1
    if iatm == 'HA' and jatm == 'H':
        if ires == FIXED_RES[0] or ires == FIXED_RES[1] or ires == FIXED_RES[2]:
            fixedhares = int(ires)
            closehres = int(jres)
        else:
            print('WARNING: The "HA" atom of the interval distance is not '
                  'the "HA" of a residue which the "CA" is fixed.')
    elif iatm == 'H' and jatm == 'HA':
        if jres == FIXED_RES[0] or jres == FIXED_RES[1] or jres == FIXED_RES[2]:
            fixedhares = int(jres)
            closehres = int(ires)
        else:
            print('WARNING: The "HA" atom of the interval distance is not '
                  'the "HA" of a residue which the "CA" is fixed.')
    else:
        print('WARNING: The hydrogen atoms of the interval distance are not '
              'a pair (HA, H) or (H, HA).')
        return None

    way = []
    if abs(fixedhares - closehres) <= 1:
        if fixedhares == FIXED_RES[0]:
            wayha = 0
        elif fixedhares == FIXED_RES[1]:
            wayha = 1
        else:
            wayha = 2

        if fixedhares == closehres:
            if wayha == 0:
                print('WARNING: The "H" atom of the first residue which the "CA" is fixed, '
                      'that is, the first residue of the protein, should not have been captured.')
                return None
            else:
                way.append(wayha - 1)
                way.append(wayha)
        elif closehres == fixedhares + 1:
            # Note: the value "3" in the first position of "way" refers to the first residue which has a fixed "CA",
            # that is, the FIXED_RES[0]-th residue.
            way.append(wayha + 1)
            way.append(wayha)
        else:
            print('WARNING: The hydrogen atom close to the "HA" of a residue which the "CA" is fixed '
                  'is in the predecessor residue, when it should be in the successor residue.')
            return None
    else:
        print('WARNING: The hydrogen atoms of the interval distance are neither '
              'in consecutive residues nor in the same residue.')
        return None

    # maps each residue index to their set of atoms
    RESIDUES = {}
    for c in df['CONSTR']:
        if c['ires'] not in RESIDUES:
            RESIDUES[c['ires']] = {}
        if c['jres'] not in RESIDUES:
            RESIDUES[c['jres']] = {}
        RESIDUES[c['ires']][c['iatm']] = c['i']
        RESIDUES[c['jres']][c['jatm']] = c['j']
    RESIDUES = {i: RESIDUES[i] for i in sorted(RESIDUES)}

    FIXED_RID = []
    i = 0
    for res in RESIDUES:
        if ((res == df['ANCHOR'][0]['res']) | (res == df['ANCHOR'][1]['res']) | (res == df['ANCHOR'][2]['res'])):
            FIXED_RID.append(i)
        i += 1

    p = []
    RESIDUES = [RESIDUES[k] for k in RESIDUES]
    # Following the NATURAL orientation of the protein
    # Note the case of way = [2, 3] is captured before as an error.
    if way[0] < way[1]:

        ## ORDER OF THE ""FIRST"" RIGID BODY'S ATOMS ##
        endmiddle = FIXED_RID[way[1]]

        # Adding atoms of the first border of the current rigid body
        if RESIDUES[FIXED_RID[way[0]]]['CA'] not in p:
            p.append(RESIDUES[FIXED_RID[way[0]]]['CA'])
        if RESIDUES[FIXED_RID[way[1]]]['CA'] not in p:
            p.append(RESIDUES[FIXED_RID[way[1]]]['CA'])
        if RESIDUES[FIXED_RID[way[1]]]['N'] not in p:
            p.append(RESIDUES[FIXED_RID[way[1]]]['N'])
        if RESIDUES[FIXED_RID[way[0]]]['C'] not in p:
            p.append(RESIDUES[FIXED_RID[way[0]]]['C'])

        p_rb = rigid_body_clockwise_order(RESIDUES, FIXED_RID[way[0]], FIXED_RID[way[1]], endmiddle)
        p = p + [atom for atom in p_rb if atom not in p]

        # Hydrogen atom to be realized by using the interval distance of "INTERV".
        if 'HA' in RESIDUES[FIXED_RID[way[1]]]:
            if RESIDUES[FIXED_RID[way[1]]]['HA'] not in p:
                p.append(RESIDUES[FIXED_RID[way[1]]]['HA'])

        ## ORDER OF THE ""SECOND"" RIGID BODY'S ATOMS ##

        way[0] += 1
        way[1] += 1

        if way[1] == 3:
            way[1] = 0
            endmiddle = len(RESIDUES)
        else:
            endmiddle = FIXED_RID[way[1]]

        if RESIDUES[FIXED_RID[way[0]]]['C'] not in p:
            p.append(RESIDUES[FIXED_RID[way[0]]]['C'])
        if RESIDUES[FIXED_RID[way[1]]]['CA'] not in p:
            p.append(RESIDUES[FIXED_RID[way[1]]]['CA'])
        if RESIDUES[FIXED_RID[way[1]]]['N'] not in p:
            p.append(RESIDUES[FIXED_RID[way[1]]]['N'])

        p_rb = rigid_body_clockwise_order(RESIDUES, FIXED_RID[way[0]], FIXED_RID[way[1]], endmiddle)
        p = p + [atom for atom in p_rb if atom not in p]

        ## ORDER OF THE ""THIRD"" RIGID BODY'S ATOMS ##

        way[0] += 1
        way[1] += 1

        if way[0] == 3:
            way[0] = 0
            endmiddle = FIXED_RID[way[1]]
        elif way[1] == 3:
            way[1] = 0
            endmiddle = len(RESIDUES)
        else:
            endmiddle = FIXED_RID[way[1]]


        if RESIDUES[FIXED_RID[way[1]]]['N'] not in p:
            p.append(RESIDUES[FIXED_RID[way[1]]]['N'])
        if 'HA' in RESIDUES[FIXED_RID[way[1]]]:
            if RESIDUES[FIXED_RID[way[1]]]['HA'] not in p:
                p.append(RESIDUES[FIXED_RID[way[1]]]['HA'])
        if RESIDUES[FIXED_RID[way[0]]]['C'] not in p:
            p.append(RESIDUES[FIXED_RID[way[0]]]['C'])
        if 'HA' in RESIDUES[FIXED_RID[way[0]]]:
            if RESIDUES[FIXED_RID[way[0]]]['HA'] not in p:
                p.append(RESIDUES[FIXED_RID[way[0]]]['HA'])

        p_rb = rigid_body_clockwise_order(RESIDUES, FIXED_RID[way[0]], FIXED_RID[way[1]], endmiddle)
        p = p + [atom for atom in p_rb if atom not in p]

    # Following the INVERSE orientation of the protein.
    # Note that way[0] is strictly greater than way[1].
    else:

        ## ORDER OF THE ""FIRST"" RIGID BODY'S ATOMS ##

        if way[0] == 3:
            way[0] = 0
            beginmiddle = len(RESIDUES)
        else:
            beginmiddle = FIXED_RID[way[0]]

        # Adding atoms of the second border, which will be the first one in the INVERSE orientation,
        # of the current rigid body
        if RESIDUES[FIXED_RID[way[0]]]['CA'] not in p:
            p.append(RESIDUES[FIXED_RID[way[0]]]['CA'])
        if RESIDUES[FIXED_RID[way[1]]]['CA'] not in p:
            p.append(RESIDUES[FIXED_RID[way[1]]]['CA'])
        if RESIDUES[FIXED_RID[way[1]]]['C'] not in p:
            p.append(RESIDUES[FIXED_RID[way[1]]]['C'])
        if RESIDUES[FIXED_RID[way[0]]]['N'] not in p:
            p.append(RESIDUES[FIXED_RID[way[0]]]['N'])

        p_rb = rigid_body_anti_clockwise_order(RESIDUES, FIXED_RID[way[1]], beginmiddle)
        p = p + [atom for atom in p_rb if atom not in p]

        # Hydrogen atom to be realized by using the interval distance of "INTERV".
        if 'HA' in RESIDUES[FIXED_RID[way[1]]]:
            if RESIDUES[FIXED_RID[way[1]]]['HA'] not in p:
                p.append(RESIDUES[FIXED_RID[way[1]]]['HA'])

        ## ORDER OF THE ""SECOND"" RIGID BODY'S ATOMS ##

        if way[0] == 0:
            way[0] = 3

        way[0] -= 1
        way[1] -= 1
        beginmiddle = FIXED_RID[way[0]]

        if way[1] < 0:
            way[1] = 2
            beginmiddle = len(RESIDUES)


        # Adding atoms of the second border, which will be the first one in the INVERSE orientation,
        # of the current rigid body
        if RESIDUES[FIXED_RID[way[0]]]['N'] not in p:
            p.append(RESIDUES[FIXED_RID[way[0]]]['N'])
        if RESIDUES[FIXED_RID[way[1]]]['CA'] not in p:
            p.append(RESIDUES[FIXED_RID[way[1]]]['CA'])
        if RESIDUES[FIXED_RID[way[1]]]['C'] not in p:
            p.append(RESIDUES[FIXED_RID[way[1]]]['C'])

        p_rb = rigid_body_anti_clockwise_order(RESIDUES, FIXED_RID[way[1]], beginmiddle)
        p = p + [atom for atom in p_rb if atom not in p]

        ## ORDER OF THE ""THIRD"" RIGID BODY'S ATOMS ##

        if way[0] == 0:
            way[0] = 3

        way[0] -= 1
        way[1] -= 1
        beginmiddle = FIXED_RID[way[0]]

        if way[1] < 0:
            way[1] = 2
            beginmiddle = len(RESIDUES)

        # Adding atoms of the second border, which will be the first one in the INVERSE orientation,
        # of the current rigid body
        if RESIDUES[FIXED_RID[way[0]]]['N'] not in p:
            p.append(RESIDUES[FIXED_RID[way[0]]]['N'])
        if 'HA' in RESIDUES[FIXED_RID[way[0]]]:
            if RESIDUES[FIXED_RID[way[0]]]['HA'] not in p:
                p.append(RESIDUES[FIXED_RID[way[0]]]['HA'])
        if RESIDUES[FIXED_RID[way[1]]]['C'] not in p:
            p.append(RESIDUES[FIXED_RID[way[1]]]['C'])
        # Adding atoms of the first border, which will be the second one in the INVERSE orientation,
        # of the current rigid body
        if 'HA' in RESIDUES[FIXED_RID[way[1]]]:
            if RESIDUES[FIXED_RID[way[1]]]['HA'] not in p:
                p.append(RESIDUES[FIXED_RID[way[1]]]['HA'])

        p_rb = rigid_body_anti_clockwise_order(RESIDUES, FIXED_RID[way[1]], beginmiddle)
        p = p + [atom for atom in p_rb if atom not in p]

    #r = []
    #a = []
    #i = 0
    #for residue in RESIDUES:
    #    for atm in residue:
    #        r.append(i)
    #        a.append(atm)
    #    i += 1
    #r = [r[p[i]] for i in range(len(r))]
    #a = [a[p[i]] for i in range(len(a))]

    #if (len(r) == len(p)) and (len(a) == len(p)):
    #    for i in range(len(p)):
    #        print('%d-th atom: %d %s %d' % (i, p[i], a[i], r[i]))

    print(p)
    return p


# builds a map that assign to each atom 'a' the atoms such that their distances to 'a' are known (the distances are stored
# as well).
def constraints(df):
    lbnd = df['INTERV']['lij']
    ubnd = df['INTERV']['uij']
    D = {}
    # note that the interval distance which is going to be discretizated is not included in the map 'D'.
    for c in df['CONSTR']:
        i = c['i']
        j = c['j']
        lij = c['lij']
        uij = c['uij']
        dij = c['lij']
        if i not in D:
            D[i] = {}
        if j not in D:
            D[j] = {}
        if lij == uij:
            D[i][j] = [dij]
            D[j][i] = [dij]
        else:
            D[i][j] = [lij, uij]
            D[j][i] = [lij, uij]
    return lbnd, ubnd, D


def solveEQ3(a, b, c, da, db, dc, dtol=1e-2, stol=1e-4):
    u = b - a
    A11 = norm(u)
    v = c - a
    A22 = norm(v)
    u = u / A11
    v = v / A22
    w = np.cross(u, v) # w perp u, v
    w = w / norm(w)
    uv = np.inner(u, v)
    A12 = A11 * uv
    A21 = A22 * uv
    # Let y = x - a, then x = y + a and y = y0*u + y1*v + y2*w
    # Using the constraints, we get
    # ||x - a|| = ||y|| = da
    # ||x - b|| = ||y - (b - a)|| = db
    # ||x - c|| = ||y - (c - a)|| = dc
    # Subtrating the square of the first from the one of two last equations, we have    
    A = [[A11, A12], [A21, A22]]
    B = [(da**2 - db**2 + A11**2)/2.0, (da**2 - dc**2 + A22**2)/2.0]
    y0, y1 = solve(A, B)
    s = da**2 - y0**2 - y1**2 - 2.0 * y0 * y1 * uv
    if s < 0 and np.abs(s) < stol:
        #print('Warning: base is almost plane (s=%g)' % s)
        s = 0
    if s < 0: # there is no solution
        # print('solveEQ3:: there is no solution (s=%g)' % s)
        return False, None, None
    
    proj_x = a + y0 * u + y1 * v # proj on the plane(a,b,c)
    y2 = np.sqrt(s)

    xpos = proj_x + y2 * w
    DA = norm(a - xpos)
    DB = norm(b - xpos)
    DC = norm(c - xpos)
    eij = np.max([np.abs(DA - da), np.abs(DB - db), np.abs(DC - dc)])
    if eij > dtol:
        raise Exception('xpos is not correct located')

    xneg = proj_x - y2 * w
    DA = norm(a - xneg)
    DB = norm(b - xneg)
    DC = norm(c - xneg)
    eij = np.max([np.abs(DA - da), np.abs(DB - db), np.abs(DC - dc)])
    if eij > dtol:
        raise Exception('xneg is not correct located')    
    
    return True, xpos, xneg


def viable(i, u, x, D, dtol=1e-2):
    for j in D[i]:
        xj = x[j]
        DIJ = norm(u - xj)

        if len(D[i][j]) == 1:
            dij = D[i][j][0]
            eij = np.abs(dij - DIJ)
            if eij > dtol:
                #print('unfeasible (i = %d, j = %d), eij=%g' % (i, j, eij))
                return False
        else:
            lij = D[i][j][0]
            uij = D[i][j][1]
            if (DIJ < lij) or (DIJ > uij):
                return False
    return True


def checkBinarySolution(s, p, x, D, dtol=1e-2):
    for i in range(4, len(s)):
        V = [j for j in D[p[i]]]
        a = x[V[0]]
        b = x[V[1]]
        c = x[V[2]]
        da = D[p[i]][V[0]]
        db = D[p[i]][V[1]]
        dc = D[p[i]][V[2]]
        _, xpos, xneg = solveEQ3(a, b, c, da, db, dc)
        if s[i] == 0:
            x[p[i]] = xpos
        else:
            x[p[i]] = xneg
    return errmax(p, x, D) < dtol
        

# writing a log file which contains just the solutions with different non-hydrogen atoms realizations.
def write_nonhlog(s, d, eijmax, rmsd, otherinfo, fname, p=None, atomvec=None, resvec=None):
    nonh_fsol = fname + '_nonH.log'
    openingOption = 'w'

    with open(nonh_fsol, openingOption) as fid:
        if (isinstance(p, list)) or (type(p).__module__ == 'numpy'):
            fid.write('order:    [ ')
            for i in p:
                fid.write('%d ' % i)
            fid.write(']\n')

        if (isinstance(atomvec, list)) or (type(atomvec).__module__ == 'numpy'):
            fid.write('atoms:    [ ')
            for i in atomvec:
                fid.write('%s ' % i)
            fid.write(']\n')

        if (isinstance(resvec, list)) or (type(resvec).__module__ == 'numpy'):
            fid.write('residues: [ ')
            for i in resvec:
                fid.write('%d ' % i)
            fid.write(']\n\n')

        fid.close()
        openingOption = 'a'

    # writing binary solutions
    if len(s) == len(d) and len(s) == len(eijmax) and len(s) == len(rmsd) and len(s) > 0:
        with open(nonh_fsol, openingOption) as fid:
            for i in range(len(s)):
                fid.write('%s %s %s s=[ ' %
                          (d[i], eijmax[i], rmsd[i]))
                for k in range(len(s[i])):
                    fid.write('%d ' % s[i][k])
                fid.write(']\n')

            fid.close()
            openingOption = 'a'


    # writing general info such as number of vertices, number of edges and running time.
    with open(nonh_fsol, openingOption) as fid:
       for info in otherinfo:
           fid.write('%s\n' % info)
           if 'nsols' in info:
               fid.write('nnonhsols %d\n' % len(s))

       fid.close()


def write_grouplog(all_s, all_d, s, d, nonhpos, intervalhapos, fname, p=None, atomvec=None, resvec=None):
    group_fsol = fname + '_group.log'
    with open(group_fsol, 'w') as fid:
        if (isinstance(p, list)) or (type(p).__module__ == 'numpy'):
            fid.write('order:    [ ')
            for i in p:
                fid.write('%d ' % i)
            fid.write(']\n')

        if (isinstance(atomvec, list)) or (type(atomvec).__module__ == 'numpy'):
            fid.write('atoms:    [ ')
            for i in atomvec:
                fid.write('%s ' % i)
            fid.write(']\n')

        if (isinstance(resvec, list)) or (type(resvec).__module__ == 'numpy'):
            fid.write('residues: [ ')
            for i in resvec:
                fid.write('%d ' % i)
            fid.write(']\n\n')


        #solsize = len(all_s[0])
        fid.write('We use a different pair of symbles to represent the bits values which are associated with the '
                  'HA and H of the discretized distance, and another one to represent the bits of other hydrogens.\n'
                  'For the HA and H from the discretized distance:\n'
                  '\t"+" means 0\n'
                  '\t"-" means 1\n'
                  'For any other hydrogens:\n'
                  '\t">" means 0\n'
                  '\t"<" means 1\n\n')

        for i in range(len(s)):
            si = s[i]
            di = d[i].split('=')[1]

            fid.write('%s s=[ ' % d[i])
            for k in range(len(si)):
                if k in nonhpos:
                    fid.write('%d ' % si[k])
                else:
                    if (intervalhapos < len(si)) and (k == intervalhapos or k == intervalhapos-1):
                        if si[k] == 0:
                            fid.write('+ ')
                        else:
                            fid.write('- ')
                    else:
                        if si[k] == 0:
                            fid.write('> ')
                        else:
                            fid.write('< ')
            fid.write(']\n')

            j = 0
            while j < len(all_s):
                #indexsdel = []
                all_dj = all_d[j].split('=')[1]

                # just binary vectors which have the same discrete interval distance are compared.
                if di == all_dj:
                    isthesame = True
                    all_sj = all_s[j]
                    for k in nonhpos:
                        if si[k] != all_sj[k]:
                            isthesame = False
                            break

                    # The atom that is realized by using a discretized distance is an alpha hydrogen.
                    if intervalhapos < len(si):
                        if si[intervalhapos] != all_sj[intervalhapos] or si[intervalhapos-1] != all_sj[intervalhapos-1]:
                            isthesame = False

                    # writing the similar solution.
                    if isthesame:
                        if si != all_sj:
                            fid.write('%s s=[ ' % all_dj)
                            for k in range(len(si)):
                                if k in nonhpos:
                                    fid.write('%d ' % all_sj[k])
                                else:
                                    if (intervalhapos < len(si)) and (k == intervalhapos or k == intervalhapos-1):
                                        if all_sj[k] == 0:
                                            fid.write('+ ')
                                        else:
                                            fid.write('- ')
                                    else:
                                        if all_sj[k] == 0:
                                            fid.write('> ')
                                        else:
                                            fid.write('< ')
                            fid.write(']\n')

                j += 1

            fid.write('\n')

        fid.close()


# writing a log file which contains just the solutions with different backbone atoms realizations.
def write_bblog(s, d, eijmax, rmsd, otherinfo, fname, p=None, atomvec=None, resvec=None):
    bb_fsol = fname + '_bb.log'
    openingOption = 'w'

    with open(bb_fsol, openingOption) as fid:
        if (isinstance(p, list)) or (type(p).__module__ == 'numpy'):
            fid.write('order:    [ ')
            for i in p:
                fid.write('%d ' % i)
            fid.write(']\n')

        if (isinstance(atomvec, list)) or (type(atomvec).__module__ == 'numpy'):
            fid.write('atoms:    [ ')
            for i in atomvec:
                fid.write('%s ' % i)
            fid.write(']\n')

        if (isinstance(resvec, list)) or (type(resvec).__module__ == 'numpy'):
            fid.write('residues: [ ')
            for i in resvec:
                fid.write('%d ' % i)
            fid.write(']\n\n')

        fid.close()
        openingOption = 'a'

    # writing binary solutions
    if len(s) == len(d) and len(s) == len(eijmax) and len(s) == len(rmsd) and len(s) > 0:
        with open(bb_fsol, openingOption) as fid:
            for i in range(len(s)):
                fid.write('%s %s %s s=[ ' %
                          (d[i], eijmax[i], rmsd[i]))
                for k in range(len(s[i])):
                    fid.write('%d ' % s[i][k])
                fid.write(']\n')

            fid.close()
            openingOption = 'a'


    # writing general info such as number of vertices, number of edges and running time.
    with open(bb_fsol, openingOption) as fid:
       for info in otherinfo:
           fid.write('%s\n' % info)
           if 'nsols' in info:
               fid.write('nbbsols . %d\n' % len(s))

       fid.close()


def write_2ndgrouplog(all_s, all_d, all_fb, s, d, fb, nonhpos, intervalhapos, fname, p=None, atomvec=None, resvec=None):
    group_fsol = fname + '_2ndgroup.log'
    with open(group_fsol, 'w') as fid:
        if (isinstance(p, list)) or (type(p).__module__ == 'numpy'):
            fid.write('order:    [ ')
            for i in p:
                fid.write('%d ' % i)
            fid.write(']\n')

        if (isinstance(atomvec, list)) or (type(atomvec).__module__ == 'numpy'):
            fid.write('atoms:    [ ')
            for i in atomvec:
                fid.write('%s ' % i)
            fid.write(']\n')

        if (isinstance(resvec, list)) or (type(resvec).__module__ == 'numpy'):
            fid.write('residues: [ ')
            for i in resvec:
                fid.write('%d ' % i)
            fid.write(']\n\n')


        #solsize = len(all_s[0])
        fid.write('We use a different pair of symbles to represent the bits values which are associated with the '
                  'HA and H of the discretized distance, and another one to represent the bits of other hydrogens.\n'
                  'For the HA and H from the discretized distance:\n'
                  '\t"+" means 0\n'
                  '\t"-" means 1\n'
                  'For any other hydrogens:\n'
                  '\t">" means 0\n'
                  '\t"<" means 1\n\n')

        this_nonhpos = list(nonhpos)
        if len(all_s) > 0 and intervalhapos < len(all_s[0]):
            this_nonhpos.remove(intervalhapos+1)

        for i in range(len(s)):
            si = s[i]
            di = d[i].split('=')[1]

            fid.write('%s s=[ ' % d[i])
            for k in range(len(si)):
                if k in nonhpos:
                    fid.write('%d ' % si[k])
                else:
                    if (intervalhapos < len(si)) and (k == intervalhapos or k == intervalhapos-1):
                        if si[k] == 0:
                            fid.write('+ ')
                        else:
                            fid.write('- ')
                    else:
                        if si[k] == 0:
                            fid.write('> ')
                        else:
                            fid.write('< ')
            fid.write(']')
            if intervalhapos < len(si):
                fid.write(' xyz%d=( %s, %s, %s )\n' % (p[intervalhapos+1], fb[i][0], fb[i][1], fb[i][2]))
            else:
                fid.write('\n')


            fbi = fb[i]
            j = 0
            while j < len(all_s):
                #indexsdel = []
                all_dj = all_d[j].split('=')[1]
                all_fbj = all_fb[j]

                # just binary vectors which have the same coords for the first atom beyond the 'HA' of the discretized
                # distance are compared.
                if fbi == all_fbj:
                    isthesame = True
                    all_sj = all_s[j]
                    for k in this_nonhpos:
                        if si[k] != all_sj[k]:
                            isthesame = False
                            break

                    # writing the similar solution.
                    if isthesame:
                        if si != all_sj:
                            fid.write('%s s=[ ' % all_dj)
                            for k in range(len(si)):
                                if k in nonhpos:
                                    fid.write('%d ' % all_sj[k])
                                else:
                                    if (intervalhapos < len(si)) and (k == intervalhapos or k == intervalhapos-1):
                                        if all_sj[k] == 0:
                                            fid.write('+ ')
                                        else:
                                            fid.write('- ')
                                    else:
                                        if all_sj[k] == 0:
                                            fid.write('> ')
                                        else:
                                            fid.write('< ')
                            fid.write(']')
                            if intervalhapos < len(all_sj):
                                fid.write(' xyz%d=( %s, %s, %s )\n' % (p[intervalhapos + 1], all_fbj[0], all_fbj[1], all_fbj[2]))
                            else:
                                fid.write('\n')

                j += 1

            fid.write('\n')

        fid.close()


def write_summarysol(all_eijmax, all_rmsd, otherinfo, df, fname):
    summaryfsol = fname + '.csv'
    pdbcode = fname.split('/')[1].split('_')[0]
    chainid = loopDefs.loc[loopDefs['pdbCode'] == pdbcode]['chainID']
    for i, item in chainid.items():
        chainid = item

    all_eijmax = [float(all_eijmax[i].split('=')[1]) for i in range(len(all_eijmax))]
    all_rmsd = [float(all_rmsd[i].split('=')[1]) for i in range(len(all_rmsd))]
    #nbbsols = len(all_eijmax) if isnonh else otherinfo[3].split(' ')[2]
    summary = {'pdb': [pdbcode + chainid + '_' + df['ANCHOR'][0]['res'] + ' (' + df['ANCHOR'][1]['res'] + ',' +
               df['ANCHOR'][2]['res'] + ')'],
               'nnodes': [otherinfo[1].split(' ')[2]],
               'nedges': [otherinfo[2].split(' ')[2]],
               'num': [otherinfo[0].split(' ')[2]],
               'idist': ['[' + '%.2f' % df['INTERV']['lij'] + ', ' + '%.2f' % df['INTERV']['uij'] + ']'],
               'nsols': [otherinfo[3].split(' ')[2]],
               'nbbsols': [len(all_eijmax)],
               'tsecs': [otherinfo[4].split(' ')[2]],
               'min_eij': [np.min(np.array(all_eijmax)) if len(all_eijmax) > 0 else '-'],
               'avg_eij': [np.mean(np.array(all_eijmax)) if len(all_eijmax) > 0 else '-'],
               'max_eij': [np.max(np.array(all_eijmax)) if len(all_eijmax) > 0 else '-'],
               'min_rmsd': [np.min(np.array(all_rmsd)) if len(all_rmsd) > 0 else '-'],
               'avg_rmsd': [np.mean(np.array(all_rmsd)) if len(all_rmsd) > 0 else '-'],
               'max_rmsd': [np.max(np.array(all_rmsd)) if len(all_rmsd) > 0 else '-'],
               }

    summary = pd.DataFrame.from_dict(summary)
    summary.to_csv(summaryfsol, index=False)


def extract_nonhsol_info(fsol, df, p):
    # map each residue to their atom index set.
    RESIDUES = {}
    for c in df['CONSTR']:
        if c['ires'] not in RESIDUES:
            RESIDUES[c['ires']] = {}
        if c['jres'] not in RESIDUES:
            RESIDUES[c['jres']] = {}
        RESIDUES[c['ires']][c['iatm']] = c['i']
        RESIDUES[c['jres']][c['jatm']] = c['j']
    RESIDUES = {i: RESIDUES[i] for i in sorted(RESIDUES)}

    # atom types of all atoms following the 'p' order.
    a = []
    inta = []
    vecres = []
    count = 0
    for residue in RESIDUES:
        ordered_atms = {k: v for k, v in sorted(RESIDUES[residue].items(), key=lambda item: item[1])}
        for atm in ordered_atms:
            a.append(atm)
            inta.append(ordered_atms[atm])
            vecres.append(count)
        count += 1
    if inta == [k for k in range(len(p))]:
        a = [a[p[i]] for i in range(len(a))]
        vecres = [vecres[p[i]] for i in range(len(vecres))]
    else:
        raise Exception('The integer representation of the atoms do not correspond to the first "n" integers.')

    alignedp = []
    for i in range(len(p)):
        if p[i] < 10:
            alignedp.append('0%d' % p[i])
        else:
            alignedp.append('%d' % p[i])
    aligneda = []
    for i in range(len(a)):
        if len(a[i]) == 1:
            aligneda.append(' %s' % a[i])
        else:
            aligneda.append('%s' % a[i])
    alignedvecres = []
    for i in vecres:
        if i < 10:
            alignedvecres.append('0%d' % i)
        else:
            alignedvecres.append('%d' % i)
    # print(p)
    print(alignedp)
    print(aligneda)
    print(alignedvecres)

    # all atoms indexes from vector 'a' of non-hydrogen atoms.
    nonhpos = []
    if len(a) == len(p):
        # non-hydrogen atoms positions at the 'p' order.
        nonhpos = [i for i in range(len(p)) if ((a[i] != 'H') & (a[i] != 'HA'))]

    # index of vector 'p' of the alpha hydrogen atom related to the discretized interval distance,
    # if the discretized interval distance is a hydrogen-hydrogen distance.
    intervalhapos = len(p)
    if df['INTERV']['jatm'] == 'HA':
        interval_ha = df['INTERV']['j']
        for i in range(len(p)):
            if p[i] == interval_ha:
                intervalhapos = i
    else:
        if df['INTERV']['jatm'] == 'H':
            raise Exception('The second INTERV atom is a hydrogen, but not a alpha hydrogen.')

    s = []
    d = []
    eijmax = []
    rmsd = []
    otherinfo = []
    with open(fsol, 'r') as fid:
        # capturing all solutions information available at the log file.
        for row in fid:
            # capturing to each solution: binary solution array, discrete interval distance, max absolute error,
            # rmsd error,
            if row[0] == 'd':
                this_s = [int(row[-i]) for i in range(4, 2 * len(p) + 3, 2)]
                this_s.reverse()
                s.append(this_s) # capturing binary solution array

                line = row.split(' ')
                d.append(line[0].replace(' ', '')) # capturing interval distance
                eijmax.append(line[1].replace(' ', ''))
                rmsd.append(line[2].replace(' ', ''))
            # capturing the number of discrete points taken from the interval, the number of vertices (atoms),
            # the number of edges (known distances), the number of solutions, the running time to solve the problem.
            else:
                otherinfo.append(row.replace('\n', ''))

        fid.close()

    # capturing the xyz coords (as strings) of the first element in 'p' beyond the 'HA' that uses the discret distance.
    firstbeyond = []
    istherefbfile = False
    if os.path.exists(fsol.split('.')[0] + '_aux.log'):
        with open(fsol.split('.')[0] + '_aux.log', 'r') as fid:
            for row in fid:
                if row[0] == 'x':
                    row = row.split(' ')
                    firstbeyond.append([row[1], row[2], row[3]])

            istherefbfile = True
            fid.close()


    print('|s|:', len(s))
    i = 0
    n = len(s)
    all_s = list(s)
    all_d = list(d)
    all_fb = list(firstbeyond)
    all_eijmax = list(eijmax)
    all_rmsd = list(rmsd)


    # FIRST FILTERING:

    # keeping just the solutions such that their non-"pruning hydrogen" part are different from the others.
    # note that 2 solutions are equal when the solution bits of the alpha hydrogen and amina hydrogen atoms from
    # the discretized interval distance are also similar.
    if istherefbfile:
        if n == len(firstbeyond):
            while i < n:
                si = s[i]
                di = d[i].split('=')[1]
                rmsdi = float(rmsd[i].split('=')[1])

                indexsdel = []
                for j in range(i + 1, n):
                    # just binary vectors which have the same discrete interval distance are compared.
                    if di == d[j].split('=')[1]:
                        isthesame = True
                        for k in nonhpos:
                            if si[k] != s[j][k]:
                                isthesame = False
                                break
                        # The atom that is realized by using a discretized distance is an alpha hydrogen.
                        if intervalhapos < len(p):
                            if si[intervalhapos] != s[j][intervalhapos] or si[intervalhapos - 1] != s[j][intervalhapos - 1]:
                                isthesame = False

                        # when two solutions have the same non-hydrogen solution, it is kept the one with the lower rmsd.
                        if isthesame:
                            if rmsdi < float(rmsd[j].split('=')[1]):
                                indexsdel.append(j)
                            else:
                                indexsdel.append(i)
                                break

                # deleting the listed solutions from the last one to the first one does not change the indexation
                # of the already evaluated solutions (the ones with index lesser than 'i').
                for j in sorted(indexsdel, reverse=True):
                    del s[j]
                    del d[j]
                    del eijmax[j]
                    del rmsd[j]
                    del firstbeyond[j]

                n -= len(indexsdel)
                if i not in indexsdel:
                    i += 1

        else:
            print('WARNING: In the FIRST SOLUTION FILTERING, the number of solutions is different from the number of R3'
                  ' coords of the first atom beyond the interval HA atom.')

    else:
        while i < n:
            si = s[i]
            di = d[i].split('=')[1]
            rmsdi = float(rmsd[i].split('=')[1])

            indexsdel = []
            for j in range(i + 1, n):
                # just binary vectors which have the same discrete interval distance are compared.
                if di == d[j].split('=')[1]:
                    isthesame = True
                    for k in nonhpos:
                        if si[k] != s[j][k]:
                            isthesame = False
                            break
                    # The atom that is realized by using a discretized distance is an alpha hydrogen.
                    if intervalhapos < len(p):
                        if si[intervalhapos] != s[j][intervalhapos] or si[intervalhapos - 1] != s[j][intervalhapos - 1]:
                            isthesame = False

                    # when two solutions have the same non-hydrogen solution, it is kept the one with the lower rmsd.
                    if isthesame:
                        if rmsdi < float(rmsd[j].split('=')[1]):
                            indexsdel.append(j)
                        else:
                            indexsdel.append(i)
                            break

            # deleting the listed solutions from the last one to the first one does not change the indexation
            # of the already evaluated solutions (the ones with index lesser than 'i').
            for j in sorted(indexsdel, reverse=True):
                del s[j]
                del d[j]
                del eijmax[j]
                del rmsd[j]

            n -= len(indexsdel)
            if i not in indexsdel:
                i += 1

    fname = fsol.split('.')[0]
    write_nonhlog(s, d, eijmax, rmsd, otherinfo, fname, p=p, atomvec=a, resvec=vecres)
    write_grouplog(all_s, all_d, s, d, nonhpos, intervalhapos, fname, p=p, atomvec=a, resvec=vecres)


    # SECOND FILTERING:

    print('nonhpos: ', nonhpos)
    print('intervalhapos: ', intervalhapos)
    print('|s|: ', len(s))

    filtered_s = list(s)
    filtered_d = list(d)
    filtered_fb = list(firstbeyond)
    filtered_eijmax = list(eijmax)
    filtered_rmsd = list(rmsd)

    # n = len(s)
    # if n == len(firstbeyond):
    #     i = 0
    #     # keeping just the solutions such that their non-hydrogen part are different from the others.
    #     while i < n:
    #         si = s[i]
    #         di = d[i].split('=')[1]
    #         rmsdi = float(rmsd[i].split('=')[1])
    #         fbi = firstbeyond[i]
    #
    #         indexsdel = []
    #         for j in range(i + 1, n):
    #             # binary vectors which have the different discret distances are also compared.
    #             isthesame = True
    #             for k in nonhpos:
    #                 if k != intervalhapos + 1:
    #                     if si[k] != s[j][k]:
    #                         isthesame = False
    #                         break
    #             # The atom that is realized by using a discretized distance is an alpha hydrogen.
    #             if intervalhapos < len(p):
    #                 # two solutions that have different R3 coords for the first atom beyond the HA of the
    #                 # discretized distance are different backbone-solutions.
    #                 if fbi != firstbeyond[j]:
    #                     isthesame = False
    #
    #             # when two solutions have the same non-hydrogen solution, it is kept the one with the lower rmsd.
    #             if isthesame:
    #                 print('di: ', di)
    #                 print('si: ', si)
    #                 print('fbi: ', fbi)
    #                 print('dj: ', d[j].split('=')[1])
    #                 print('sj: ', s[j])
    #                 print('fbj: ', firstbeyond[j])
    #                 if rmsdi < float(rmsd[j].split('=')[1]):
    #                     indexsdel.append(j)
    #                 else:
    #                     indexsdel.append(i)
    #                     break
    #
    #         # deleting the listed solutions from the last one to the first one does not change the indexation
    #         # of the already evaluated solutions (the ones with index lesser than 'i').
    #         for j in sorted(indexsdel, reverse=True):
    #             del s[j]
    #             del d[j]
    #             del eijmax[j]
    #             del rmsd[j]
    #             del firstbeyond[j]
    #
    #         n -= len(indexsdel)
    #         if i not in indexsdel:
    #             i += 1

    if istherefbfile:
        nonhpos_mfb = list(nonhpos)
        if intervalhapos < len(p):
            nonhpos_mfb.remove(intervalhapos+1)

        if intervalhapos < len(p):
            n = len(s)
            if n == len(firstbeyond):
                i = 0
                # keeping just the solutions such that their non-hydrogen part are different from the others.
                while i < n:
                    si = s[i]
                    di = d[i].split('=')[1]
                    rmsdi = float(rmsd[i].split('=')[1])
                    fbi = firstbeyond[i]

                    indexsdel = []
                    for j in range(i + 1, n):
                        # just binary vectors which have the same coords for the first atom beyond the 'HA' of the discretized
                        # distance are compared.
                        if fbi == firstbeyond[j]:
                            isthesame = True
                            for k in nonhpos_mfb:
                                if si[k] != s[j][k]:
                                    isthesame = False
                                    break

                            # when two solutions have the same non-hydrogen solution, it is kept the one with the lower rmsd.
                            if isthesame:
                                if rmsdi < float(rmsd[j].split('=')[1]):
                                    indexsdel.append(j)
                                else:
                                    indexsdel.append(i)
                                    break

                    # deleting the listed solutions from the last one to the first one does not change the indexation
                    # of the already evaluated solutions (the ones with index lesser than 'i').
                    for j in sorted(indexsdel, reverse=True):
                        del s[j]
                        del d[j]
                        del eijmax[j]
                        del rmsd[j]
                        del firstbeyond[j]

                    n -= len(indexsdel)
                    if i not in indexsdel:
                        i += 1
            else:
                print('WARNING: In the SECOND SOLUTION FILTERING, the number of solutions is different from the number '
                      'of R3 coords of the first atom beyond the interval HA atom.')
    else:
        print('WARNING: In the SECOND SOLUTION FILTERING, the file containing the R3 coords of the first atom beyond '
              'the interval HA atom does not exist.')

    print('|s|: ', len(s))
    print('|fb|: ', len(firstbeyond))
    #
    # hposset = {i for i in range(len(p))} - set(nonhpos)
    # if intervalhapos < len(p):
    #     hposset = hposset - {intervalhapos, intervalhapos-1}
    # isnonh = True if len(hposset) > 0 else False

    write_bblog(s, d, eijmax, rmsd, otherinfo, fname, p=p, atomvec=a, resvec=vecres)
    if istherefbfile:
        write_2ndgrouplog(filtered_s, filtered_d, filtered_fb, s, d, firstbeyond, nonhpos, intervalhapos, fname, p=p, atomvec=a, resvec=vecres)
    # write_summarysol(all_eijmax, all_rmsd, otherinfo, df, fname)
    write_summarysol(eijmax, rmsd, otherinfo, df, fname)  # , isnonh=isnonh)


def errmax(p, x, D):
    eijMax = 0
    for k in range(3, len(p)):
        i = p[k]
        xi = x[i]
        for j in D[i]:
            xj = x[j]
            DIJ = norm(xi - xj)
            if len(D[i][j]) == 1:
                dij = D[i][j][0]
                eij = np.abs(dij - DIJ)
                if eij > eijMax:
                    eijMax = eij
            elif len(D[i][j]) > 1:
                iserror = False
                if DIJ < D[i][j][0]:
                    eij = D[i][j][0] - DIJ
                    iserror = True
                if DIJ > D[i][j][1]:
                    eij = DIJ - D[i][j][1]
                    iserror = True
                if iserror:
                    if eij > eijMax:
                        eijMax = eij
            else:
                raise Exception('errmax: ')
    return eijMax


def distance_range(i, a, b, c, D):
    # range of distance between i and c
    dab = D[a][b][0] if b in D[a] else D[b][a][0]
    dac = D[a][c][0] if c in D[a] else D[c][a][0]
    dbc = D[b][c][0] if c in D[b] else D[c][b][0]
    dax = D[a][i][0] if i in D[a] else D[i][a][0]
    dbx = D[b][i][0] if i in D[b] else D[i][b][0]
    Cx = (dab**2 + dac**2 - dbc**2) / (2 * dab)
    Cy = np.sqrt(dac**2 - Cx**2)
    A = np.array([0, 0, 0], dtype=float)
    B = np.array([dab, 0,  0], dtype=float)
    C = np.array([Cx, Cy, 0], dtype=float)
    dAB = distance(A, B)
    dAC = distance(A, C)
    dBC = distance(B, C)
    if np.max([np.abs(dAB - dab), np.abs(dAC - dac), np.abs(dBC - dbc)]) > 1e-8:
        raise Exception('A,B,C are not correct located')
    Xx = (dab**2 + dax**2 - dbx**2) / (2*dab)
    Xy = np.sqrt(dax**2 - Xx**2)
    Xmin = np.array([Xx, Xy, 0], dtype=float)
    dAX = distance(A, Xmin)
    dBX = distance(B, Xmin)
    if np.max([np.abs(dAX - dax), np.abs(dBX - dbx)]) > 1e-8:
        raise Exception('Xmin is not correct located')
    Xmax = np.array([Xx, -Xy, 0], dtype=float)

    lij = distance(C, Xmin) + 1E-3
    uij = distance(C, Xmax) - 1E-3
    return lij, uij


def bp(p, i, x, D, s, df, fid, num, chooseNeighbours=True, fidaux=None):
    if i == len(p):
        df['nsols'] += 1
        ditv = norm(x[int(df['INTERV']['i'])]-x[int(df['INTERV']['j'])])
        eijMax = errmax(p, x, D)
        rval, _, _ = rmsd(x, df['xsol'])
        fid.write('d_%d_%d=%.12f eijMax=%.3E rmsd=%.3E s=[ ' %
                  (int(df['INTERV']['i']), int(df['INTERV']['j']), ditv, eijMax, rval))
        for k in range(len(s)):
            fid.write('%d ' % s[k])
        fid.write(']\n')

        if fidaux != None:
            # the index of the HA of the discretized distance
            k = p.index(int(df['INTERV']['j']))
            fidaux.write('xyz%d=( %f %f %f )\n' % (p[k+1], x[p[k+1]][0], x[p[k+1]][1], x[p[k+1]][2]))
        return

    V = [j for j in D[p[i]]] # adjacent antecessors

    abc = []
    nabc = 3

    # capturing the most recently realized neighbours of the i-th element such that their distance to the i-th element
    # is exact.
    if chooseNeighbours:
        for j in range(1, len(V)):
            if len(D[p[i]][V[-j]]) == 1:
                abc.append(V[-j])
            if len(abc) >= nabc:
                break
        if len(abc) < nabc:
            if len(D[p[i]][V[0]]) == 1:
                abc.append(V[0])
    # to run the Labiak-Lavor-Souza experiment
    else:
        for j in range(0, len(V)):
            if len(D[p[i]][V[j]]) == 1:
                abc.append(V[j])
            if len(abc) >= nabc:
                break

    # The realization of the current vertex is only possible if it is known at least 2 exact distances to previous vertices.
    if len(abc) >= 2:
        a = x[abc[0]]
        b = x[abc[1]]
        da = D[p[i]][abc[0]][0]
        db = D[p[i]][abc[1]][0]

        # realizing the i-th atom by using 3 exact distances.
        if len(abc) >= 3:
            c = x[abc[2]]
            dc = D[p[i]][abc[2]][0]

            solved, xpos, xneg = solveEQ3(a, b, c, da, db, dc)
            if not solved:
                return

            # try xpos solution
            if viable(p[i], xpos, x, D):
                s[i] = 0
                x[p[i]] = xpos
                bp(p, i+1, x, D, s, df, fid, num, chooseNeighbours=chooseNeighbours, fidaux=fidaux)
            # try xneg solution
            if viable(p[i], xneg, x, D) and np.linalg.norm(xpos - xneg) != 0.0:
                s[i] = 1
                x[p[i]] = xneg
                bp(p, i+1, x, D, s, df, fid, num, chooseNeighbours=chooseNeighbours, fidaux=fidaux)

        # realizing the i-th atom by using 2 exact distances and 1 interval distance.
        else:
            if int(df['INTERV']['i']) == p[i] or int(df['INTERV']['j']) == p[i]:
                abc.append(int(df['INTERV']['i']))
                lbnd = float(df['INTERV']['lij'])
                ubnd = float(df['INTERV']['uij'])
            else:
                raise Exception('There is an atom other than the ones in INTERV which cannot be realized with '
                                '3 exact distances.')

            c = x[abc[2]]
            DC = [dc for dc in np.linspace(lbnd, ubnd, num=num, endpoint=True, dtype=float)]
            for dc in DC:

                solved, xpos, xneg = solveEQ3(a, b, c, da, db, dc)
                if not solved:
                    continue

                # try xpos solution
                if viable(p[i], xpos, x, D):
                    s[i] = 0
                    x[p[i]] = xpos
                    bp(p, i + 1, x, D, s, df, fid, num, chooseNeighbours=chooseNeighbours, fidaux=fidaux)
                # try xneg solution
                if viable(p[i], xneg, x, D):
                    s[i] = 1
                    x[p[i]] = xneg
                    bp(p, i + 1, x, D, s, df, fid, num, chooseNeighbours=chooseNeighbours, fidaux=fidaux)
    else:
        raise Exception('The %d-th element of the vertex order cannot be realized: there is less than 2 exact distances '
                        'to previous vertices.')


def bpl(p, lbnd, ubnd, D, df, fid, num=100, chooseNeighbours=True, fidaux=None):
    df['nsols'] = 0

    x = np.zeros((len(p), 3), dtype=float)

    # set base: the vertex order uses the 3 anchors as its first 3 elements
    anchors = [int(df['ANCHOR'][0]['i']), int(df['ANCHOR'][1]['i']), int(df['ANCHOR'][2]['i'])]
    if (p[0] in anchors) and (p[1] in anchors) and (p[2] in anchors): # base of non-hydrogen order
        p3s = [p[0], p[1], p[2]]
        xp3s = []
        for i in range(len(p3s)):
            for j in range(len(p3s)):
                if p[i] == df['ANCHOR'][j]['i']:
                    xp3s.append(df['ANCHOR'][j]['x'])

        for i in range(len(p3s)):
            x[p[i]] = xp3s[i]

        ishydro = False
    # set base: the interval distance to be discretized is a hydrogen-hydrogen distance, then the vertex order uses
    # 2 anchors and an inner vertex as its first 3 elements
    elif (df['INTERV']['iatm'] == 'H' or df['INTERV']['iatm'] == 'HA') and (df['INTERV']['jatm'] == 'H' or df['INTERV']['jatm'] == 'HA'): # base of hydrogen order

        # The 'HA' atom of one of the exterior residues participates in the 'INTERV' distance of our proposed vertex order.
        # Thus, we organize the 'INTERV' line from the input files '.dat' in a way that the 'HA' atom is set as the
        # second atom. So:
        #   * If the first atom of the 'INTERV' line is a 'H' such that its residue is the same residue of
        #     the 'HA' (the second atom of 'INTERV' line), then it means the proposed vertex order follows a clockwise
        #     orientation, starting at the rigid body that 'contains' the 'H' atom.
        #   * Otherwise, if the 'INTERV' line is a 'H' such that its residue is the residue that is the successor of
        #     the residue that contains the 'HA' (the second atom of 'INTERV' line), then we know the proposed vertex
        #     order follows an anticlockwise orientation, starting at the rigid body that 'contains' the 'H' atom.

        # second border of the first rigid body to be realized.
        if df['INTERV']['jatm'] == 'HA' and int(df['INTERV']['jres']) == int(df['ANCHOR'][0]['res']):
            bd2 = 0
        elif df['INTERV']['jatm'] == 'HA' and int(df['INTERV']['jres']) == int(df['ANCHOR'][1]['res']):
            bd2 = 1
        elif df['INTERV']['jatm'] == 'HA' and int(df['INTERV']['jres']) == int(df['ANCHOR'][2]['res']):
            bd2 = 2
        else:
            raise Exception('bpl: the base cannot be set because the second "INTERV" atom is not from a'
                            '"ANCHOR" residue.')

        # first border of the first rigid body to be realized.
        if (df['INTERV']['iatm'] == 'H') and (int(df['INTERV']['ires']) == int(df['INTERV']['jres'])):
            bd1 = bd2 - 1
            if bd1 < 0:
                bd1 = 2
        elif (df['INTERV']['iatm'] == 'H') and (int(df['INTERV']['ires']) == int(df['INTERV']['jres']) + 1):
            bd1 = bd2 + 1
            if bd1 > 2:
                bd1 = 0
        else:
            raise Exception('bpl: the base cannot be set because the first "INTERV" atom is neither the "H" '
                  'of the following residue nor the "H" of the same residue of the "ANCHOR" "HA".')

        # coords of the first atom in the hydrogen order
        x[p[0]] = df['ANCHOR'][bd1]['x']

        # coords of the second atom in the hydrogen order
        x[p[1]] = df['ANCHOR'][bd2]['x']

        otheranchor = [0, 1, 2]
        otheranchor.remove(bd1)
        otheranchor.remove(bd2)
        otheranchor = otheranchor[0]

        lij, uij = distance_range(p[2], int(df["ANCHOR"][bd1]['i']), int(df["ANCHOR"][bd2]['i']),
                                  int(df["ANCHOR"][otheranchor]['i']), D)
        da = D[p[2]][int(df["ANCHOR"][bd1]['i'])][0] if int(df["ANCHOR"][bd1]['i']) in D[p[2]] \
            else D[int(df["ANCHOR"][bd1]['i'])][p[2]][0]
        db = D[p[2]][int(df["ANCHOR"][bd2]['i'])][0] if int(df["ANCHOR"][bd2]['i']) in D[p[2]] \
            else D[int(df["ANCHOR"][bd2]['i'])][p[2]][0]
        dc = (lij + uij)/2
        solved, xpos, xneg = solveEQ3(df["ANCHOR"][bd1]['x'], df["ANCHOR"][bd2]['x'],
                                  df["ANCHOR"][otheranchor]['x'], da, db, dc)
        # coords of the third atom in the hydrogen order
        if solved:
            x[p[2]] = xpos
        else:
            raise Exception('bpl: The realization of the THIRD vertex failed: the intersection of 3 spheres is empty.')

        ishydro = True
    else:
        raise Exception('bpl: The "p" order is neither our proposed vertex order nor the Labiak-Lavor-Souza order.')

    # solution binary representation
    s = np.zeros(len(p), dtype=int) 

    d03 = D[p[3]][p[0]][0]
    d23 = D[p[3]][p[2]][0]

    if ishydro:
        d13 = D[p[3]][p[1]][0]

        # only positive oriented branch
        solved, xpos, xneg = solveEQ3(x[p[0]], x[p[1]], x[p[2]], d03, d13, d23)
        if not solved:
            raise Exception('bpl::the first three vertices are not a clique which respects '
                            'the strict triangular inequality')

        x[p[3]] = xpos
        s[3] = 0
        bp(p, 4, x, D, s, df, fid, num, chooseNeighbours=chooseNeighbours, fidaux=fidaux)

        # TODO Consider to use only one root (first branch)
        x[p[3]] = xneg
        s[3] = 1
        bp(p, 4, x, D, s, df, fid, num, chooseNeighbours=chooseNeighbours, fidaux=fidaux)
    else:
        D13 = [d13 for d13 in np.linspace(lbnd, ubnd, num=num, endpoint=True, dtype=float)]

        for d13 in D13:
            # only positive oriented branch
            solved, xpos, xneg = solveEQ3(x[p[0]], x[p[1]], x[p[2]], d03, d13, d23)
            if not solved:
                raise Exception('ibpl::the first system could not be solved')

            x[p[3]] = xpos
            s[3] = 0
            bp(p, 4, x, D, s, df, fid, num, chooseNeighbours=chooseNeighbours, fidaux=fidaux)

            # TODO Consider to use only one root (first branch)
            x[p[3]] = xneg
            s[3] = 1
            bp(p, 4, x, D, s, df, fid, num, chooseNeighbours=chooseNeighbours, fidaux=fidaux)


def run_bpl(fdat: str, num: int):
    if os.path.exists(fdat):
        df = read_dat(fdat)
        print('num: ', num)
        df['xsol'], _ = read_sol(fdat.replace('.dat', '.sol'))
        lbnd, ubnd, D = constraints(df)

        is_h_order = (df['INTERV']['iatm'] == 'H' or df['INTERV']['iatm'] == 'HA') and (df['INTERV']['jatm'] == 'H' or df['INTERV']['jatm'] == 'HA')

        auxfdat = fdat.replace('.dat', '_%d_aux.log' % num)
        if is_h_order:
            p = hydrogen_order(df)
            D = cleanAndReorganizeD(p, D)
            isLLS_order = False
            fid2 = open(auxfdat, 'w')
        else:
            p = Labiak_Lavor_Souza_order(df)
            # adding the hydrogen atoms at the end of the Labiak-Lavor-Souza order
            p = p + list(set([i for i in range(len(D))]) - set(p))
            D = cleanD(p, D)
            isLLS_order = True
            fid2 = None

        flog = fdat.replace('.dat', '_%d.log' % num)
        fid = open(flog, 'w')
        print('Writing ' + flog)

        tic = time.time()
        bpl(p, lbnd, ubnd, D, df, fid, num, chooseNeighbours=not isLLS_order, fidaux=fid2)
        toc = time.time() - tic

        nedges = 0
        for i in D:
            nedges += len(D[i])

        fid.write('num ..... %d\n' % num)
        fid.write('nnodes .. %d\n' % len(p))
        fid.write('nedges .. %d\n' % nedges)
        fid.write('nsols ... %d\n' % df['nsols'])
        fid.write('tsecs ... %f\n' % toc)
        
        fid.close()
        if fid2 != None:
            fid2.close()

        extract_nonhsol_info(fdat.split('.')[0] + '_' + str(num) + '.log', df, p)


def test_run_bpl():
    num = 1000
    # fdir = 'DATA_LOOP_12'
    # fname = '1qop.dat'
    fdir = 'DATA_LOOP_08'
    fname = '1cru.dat'
    fdat = os.path.join(fdir, fname)
    run_bpl(fdat, num)


if __name__ == '__main__':
    test_run_bpl()
    
    # # default parameters
    # num = 1000
    # fdat = 'DATA_LOOP_12/1qop.dat'

    # # read input
    # if len(sys.argv) > 1:
    #     fdat = sys.argv[1]
    # if len(sys.argv) > 2:
    #     num = int(sys.argv[2])

    # # create a folder results. Inside this folder, there will be
    # # the folders DATA_LOOP_04, DATA_LOOP_08, DATA_LOOP_12 that will
    # # contain the log files
    # if not os.path.exists('results'):
    #     os.mkdir('results')
    #     os.mkdir('results/DATA_LOOP_04')
    #     os.mkdir('results/DATA_LOOP_08')
    #     os.mkdir('results/DATA_LOOP_12')
    
    # df = read_dat(fdat)
    # print('num: ', num)
    # df['xsol'], _ = read_sol(fdat.replace('.dat', '.sol'))
    
    # lbnd, ubnd, D = constraints(df)
    # p = hydrogen_order(df)

    # auxfdat = fdat.replace('.dat', '_%d_aux.log' % num)
    # if p == None:
    #     p = Labiak_Lavor_Souza_order(df)
    #     # adding the hydrogen atoms at the end of the Labiak-Lavor-Souza order
    #     p = p + list(set([i for i in range(len(D))]) - set(p))
    #     D = cleanD(p, D)
    #     isLLS_order = True
    #     if os.path.exists(auxfdat):
    #         os.remove(auxfdat)
    #     if os.path.exists(auxfdat.replace('_aux.log', '_2ndgroup.log')):
    #         os.remove(auxfdat.replace('_aux.log', '_2ndgroup.log'))
    #     fid2 = None
    # else:
    #     D = cleanAndReorganizeD(p, D)
    #     isLLS_order = False
    #     fid2 = open(auxfdat, 'w')
    
    # # # flog should place the log file in the results folder
    # # parts = os.path.split(fdat)
    # # # taking the folder that stores the data file
    # # flog = os.path.join('results', parts[-2])
    # # # defining the log file name
    # # flog = os.path.join(flog, parts[-1].replace('.dat', '_%d.log' % num))

    # flog = fdat.replace('.dat', '_%d.log' % num)
    # fid = open(flog, 'w')
    # print('Writing ' + flog)

    # tic = time.time()
    # bpl(p, lbnd, ubnd, D, df, fid, num, chooseNeighbours=not isLLS_order, fidaux=fid2)
    # toc = time.time() - tic

    # nedges = 0
    # for i in D:
    #     nedges += len(D[i])

    # fid.write('num ..... %d\n' % num)
    # fid.write('nnodes .. %d\n' % len(p))
    # fid.write('nedges .. %d\n' % nedges)
    # fid.write('nsols ... %d\n' % df['nsols'])
    # fid.write('tsecs ... %f\n' % toc)
    
    # fid.close()
    # if fid2 != None:
    #     fid2.close()

    # extract_nonhsol_info(fdat.split('.')[0] + '_' + str(num) + '.log', df, p)
