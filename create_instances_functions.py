import pandas as pd
import os
import sys
import numpy as np


def distance(xi, xj):
    return np.sqrt((xi[0] - xj[0])**2 + (xi[1] - xj[1])**2 + (xi[2] - xj[2])**2)


def distance_range(i, a, b, c, V):
    # range of distance between i and c
    X = V[i]['XYZ']
    A = V[a]['XYZ']
    B = V[b]['XYZ']
    C = V[c]['XYZ']
    dab = distance(A, B)
    dac = distance(A, C)
    dbc = distance(B, C)
    dax = distance(A, X)
    dbx = distance(B, X)
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


def generate_interval_distH(d, interval_len, rng):
    deltaij = rng.uniform(0, interval_len)
    lij = max(0, d - deltaij)
    uij = d + (interval_len - deltaij)

    return lij, uij


def checkResidues(RESIDUES, resSeqFixed=None):
    if not ('H' in RESIDUES[resSeqFixed]) and not ('HA' in RESIDUES[resSeqFixed]):
        if not ('H' in RESIDUES[resSeqFixed+1]) and not ('HA' in RESIDUES[resSeqFixed]):
            raise Exception('The residue of the second fixed "CA", that is, the %d-th residue, does not have its '
                            'both "H" and "HA".' % resSeqFixed)
            print(RESIDUES[resSeqFixed])
    for resSeq in RESIDUES:
        if len(RESIDUES[resSeq]) < 3:
            print(RESIDUES[resSeq])
            raise Exception('The %d-th residue is not well defined.' % resSeq)
    return True


def removeAlphaHydrogen(df, resSeqFixed):
    dfwHA = df.loc[(df['resSeq'] != resSeqFixed) & ((df['name'] == 'HA') | (df['name'] == 'HA2'))]
    if not dfwHA.empty:
        df.drop(dfwHA.index, inplace=True)

    return df


def removeAminoHydrogen(df, resSeqFixed, isnatorder):
    auxhres = 0 if isnatorder else 1
    dfwH = df.loc[(df['resSeq'] != (resSeqFixed + auxhres)) & ((df['name'] == 'H') | (df['name'] == 'H1'))]

    if not dfwH.empty:
        df.drop(dfwH.index, inplace=True)

    return df


def removeAllHydrogen(df):
    dfwHandHA = df.loc[((df['name'] == 'H') | (df['name'] == 'H1') | (df['name'] == 'HA') | (df['name'] == 'HA2'))]

    if not dfwHandHA.empty:
        df.drop(dfwHandHA.index, inplace=True)

    return df


def createCloseHColumn(df, resSeqFixed):
    # creates a new column which indicates if a hydrogen is close to the alpha hydrogen (of a residue of
    # fixed CA) which was chosen to have the interval distance.
    rowfHA = df.loc[(df['resSeq'] == resSeqFixed) & ((df['name'] == 'HA') | (df['name'] == 'HA2'))]
    xyzfHA = (float(rowfHA.x), float(rowfHA.y), float(rowfHA.z))
    closeH = df.apply(lambda row:
                      True if ((distance((float(row.x), float(row.y), float(row.z)), xyzfHA) <= 5.0) & (row.element == 'H')) else False, axis=1)

    df['closeH'] = closeH
    return closeH


def readResidues(df, resSeqFixed = None, istoremoveAllH = False, istoremoveH = False, isnatorder = None): # adaptado aos hidrogenios
    RESIDUES = {}

    def ajustHydrogenNames(name):
        if name == 'H1':
            name = 'H'
        if name == 'HA2':
            name = 'HA'
        return name

    # putting hydrogen atoms names in a default name
    df['name'] = df['name'].apply(lambda name: ajustHydrogenNames(name))

    if istoremoveAllH:
        df = removeAllHydrogen(df)
        istoremoveH = False

    # Removing all alpha hydrogen atoms except the "HA" from the residue 'resSeqFixed', which has a fixed "CA"
    if istoremoveH and isnatorder != None:
        df = removeAlphaHydrogen(df, resSeqFixed)
        df = removeAminoHydrogen(df, resSeqFixed, isnatorder)

    # group N, C, CA, H and HA of each residue
    for _, row in df.iterrows():

        resSeq = row['resSeq']
        if resSeq not in RESIDUES:
            RESIDUES[resSeq] = {}

        atom = {'nid': None, 'XYZ': (
            float(row.x), float(row.y), float(row.z)), 'resSeq': resSeq, 'name': row['name']}
        RESIDUES[resSeq][row['name']] = atom

    #checkResidues(RESIDUES, resSeqFixed)

    # convert from dict to list of RESIDUES
    nid = 0  # node id
    for resSeq in sorted(RESIDUES):
        RESIDUES[resSeq]['resSeq'] = resSeq
        RESIDUES[resSeq]['N']['nid'] = nid
        RESIDUES[resSeq]['CA']['nid'] = nid + 1
        RESIDUES[resSeq]['C']['nid'] = nid + 2
        nid += 3

        ihydro = 0
        if 'H' in RESIDUES[resSeq]:
            RESIDUES[resSeq]['H']['nid'] = nid
            ihydro += 1

        if 'HA' in RESIDUES[resSeq]:
            RESIDUES[resSeq]['HA']['nid'] = nid + ihydro
            ihydro += 1

        nid += ihydro

    RESIDUES = [RESIDUES[resSeq] for resSeq in sorted(RESIDUES)]

    return RESIDUES, nid


def writeRIGID(fid, k, R, V, XYZ, verbose=False):
    row = '\nINFO: COLUMN RES_I RES_J  ATM_I ATM_J IDX_I IDX_J         LIJ             UIJ'
    if verbose: print(row)
    fid.write(row + '\n')
    for i in range(len(R)):
        for j in range(i+1, len(R)):
            xi = XYZ[R[i]]
            xj = XYZ[R[j]]
            vi = V[R[i]]
            vj = V[R[j]]
            dij = distance(xi, xj)
            row = 'DATA: RIGID%d %5d %5d' % (k, vi['resSeq'], vj['resSeq'])
            row += '%7s %5s' % (vi['name'], vj['name'])
            row += '%6d %5d' % (vi['nid'] + 1, vj['nid'] + 1)
            row += '%16.8f' % dij
            row += '%16.8f' % dij
            if verbose: print(row)
            fid.write(row + '\n')


# Writing the distances between hydrogens which belong to the SAME rigid body.
def writeHYDRO(fid, k, H, V, XYZ, itvAtomIdi, itvAtomIdj, interval_len, rng, verbose=False):
    row = '\nINFO: COLUMN RES_I RES_J  ATM_I ATM_J IDX_I IDX_J         LIJ             UIJ'
    if verbose: print(row)
    fid.write(row + '\n')

    for i in range(len(H)):
        # writing distances between the current hydrogen atom and its non-hydrogen neighbours.
        for j in range(1, len(H[i])):
            xi = XYZ[H[i][j]]
            xj = XYZ[H[i][0]]
            vi = V[H[i][j]]
            vj = V[H[i][0]]
            dij = distance(xi, xj)
            row = 'DATA: HYDRORIG%d %5d %5d' % (k, vi['resSeq'], vj['resSeq'])
            row += '%7s %5s' % (vi['name'], vj['name'])
            row += '%6d %5d' % (vi['nid'] + 1, vj['nid'] + 1)
            row += '%16.8f' % dij
            row += '%16.8f' % dij
            if verbose: print(row)
            fid.write(row + '\n')

        # writing distances between the current hydrogen atom and other hydrogen atoms from the same rigid body and which
        # has not yet been written.
        for j in range(i + 1, len(H)):
            xi = XYZ[H[i][0]]
            xj = XYZ[H[j][0]]
            dij = distance(xi, xj)
            if dij <= 5.0:
                lij, uij = generate_interval_distH(dij, interval_len, rng)
                # The NMR experiment detects hydrogen-hydrogen distances bounded superiorly by 5.0 angstroms.
                if uij > 5.0:
                    lij = lij - (uij - 5.0)
                    uij = 5.0

                vi = V[H[i][0]]
                vj = V[H[j][0]]
                if (vi['nid']+1 == itvAtomIdi) & (vj['nid']+1 == itvAtomIdj) or \
                        (vj['nid']+1 == itvAtomIdi) & (vi['nid']+1 == itvAtomIdj):
                    continue
                row = 'DATA: HYDRORIG%d %5d %5d' % (k, vi['resSeq'], vj['resSeq'])
                row += '%7s %5s' % (vi['name'], vj['name'])
                row += '%6d %5d' % (vi['nid'] + 1, vj['nid'] + 1)
                row += '%16.8f' % lij
                row += '%16.8f' % uij
                if verbose: print(row)
                fid.write(row + '\n')


# Writing the distances between hydrogens which belong to DIFFERENT rigid bodies.
def writeCROSSHYDRO(fid, k1, k2, Ha, Hb, posHAa_b, posHAb_a, V, XYZ, itvAtomIdi, itvAtomIdj, interval_len, rng, verbose=False):
    row = '\nINFO: COLUMN RES_I RES_J  ATM_I ATM_J IDX_I IDX_J         LIJ             UIJ'
    if verbose: print(row)
    fid.write(row + '\n')

    # Removing the alpha hydrogen linked to the fixed alpha carbon that is common to both rigid bodies.
    removed = False
    if posHAa_b < len(Ha) and posHAb_a < len(Hb):
        if V[Ha[posHAa_b][0]] == V[Hb[posHAb_a][0]]:
            el_a = Ha.pop(posHAa_b)
            el_b = Hb.pop(posHAb_a)
            removed = True
        else:
            print('WARNING: The alpha hydrogen linked to the fixed alpha carbon that is common to "Ha" and "Hb" is not in a'
                  'correct position in either "Ha" or "Hb".')
    else:
        print('WARNING: The fixed alpha carbon which is common to both residues does not have alpha hydrogen attached to it.')

    # writing distances between each hydrogen atom from the rigid body 'a' and each hydrogen atom from the rigid body 'b'.
    for i in range(len(Ha)):
        for j in range(len(Hb)):
            xi = XYZ[Ha[i][0]]
            xj = XYZ[Hb[j][0]]
            dij = distance(xi, xj)
            if dij <= 5.0:
                lij, uij = generate_interval_distH(dij, interval_len, rng)
                # The NMR experiment detects hydrogen-hydrogen distances bounded superiorly by 5.0 angstroms.
                if uij > 5.0:
                    lij = lij - (uij - 5.0)
                    uij = 5.0

                vi = V[Ha[i][0]]
                vj = V[Hb[j][0]]
                if (vi['nid']+1 == itvAtomIdi) & (vj['nid']+1 == itvAtomIdj) or \
                        (vj['nid']+1 == itvAtomIdi) & (vi['nid']+1 == itvAtomIdj):
                    continue
                row = 'DATA: CROSSHYDRORIG_%d_%d %5d %5d' % (k1, k2, vi['resSeq'], vj['resSeq'])
                row += '%7s %5s' % (vi['name'], vj['name'])
                row += '%6d %5d' % (vi['nid'] + 1, vj['nid'] + 1)
                row += '%16.8f' % lij
                row += '%16.8f' % uij
                if verbose: print(row)
                fid.write(row + '\n')

    if removed:
        Ha.insert(posHAa_b, el_a)
        Hb.insert(posHAb_a, el_b)


def writeAllCROSSHYDRO(fid, H1: list, H2: list, H3: list, isthereHA123: list, V: list, XYZ: list, itvAtomIdi: int, itvAtomIdj: int, interval_len: int, rng: np.random.RandomState):
    # hydrogen neighbourhood between atoms from the FIRST and SECOND rigid bodies
    H = [H1, H2, H3]
    if isthereHA123[1]:
        posHAa_b = len(H[0]) - 1
        posHAb_a = 0
    else:
        posHAa_b = len(H[0])
        posHAb_a = len(H[1])
    writeCROSSHYDRO(fid, 0, 1, H[0], H[1], posHAa_b, posHAb_a, V, XYZ, itvAtomIdi, itvAtomIdj,
                interval_len, rng)

    # hydrogen neighbourhood between atoms from the SECOND and THIRD rigid bodies
    if isthereHA123[2]:
        posHAa_b = len(H[1]) - 1
        posHAb_a = 0
    else:
        posHAa_b = len(H[1])
        posHAb_a = len(H[2])
    writeCROSSHYDRO(fid, 1, 2, H[1], H[2], posHAa_b, posHAb_a, V, XYZ, itvAtomIdi, itvAtomIdj,
                    interval_len, rng)

    # hydrogen neighbourhood between atoms from the THIRD and FIRST rigid bodies
    if isthereHA123[0]:
        posHAa_b = len(H[2]) - 1
        posHAb_a = 0
    else:
        posHAa_b = len(H[2])
        posHAb_a = len(H[0])
    writeCROSSHYDRO(fid, 0, 2, H[0], H[2], posHAb_a, posHAa_b, V, XYZ, itvAtomIdi, itvAtomIdj,
                    interval_len, rng)


def writeLNK(fid, R, V, i, j, XYZ, verbose=False):
    xi = XYZ[i]
    xj = XYZ[j]
    vi = V[i]
    vj = V[j]
    row = 'DATA: LINKS  %5d %5d' % (vi['resSeq'], vj['resSeq'])
    row += '%7s %5s' % (vi['name'], vj['name'])
    row += '%6d %5d' % (vi['nid'] + 1, vj['nid'] + 1)
    dij = distance(xi, xj)
    row += '%8.5f' % dij
    row += '%8.5f' % dij
    if verbose: print(row)
    fid.write(row + '\n')