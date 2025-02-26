import pandas as pd
import os
import sys
import numpy as np
import create_instances_functions as crf

def create_hclp_instances(folder: str, interval_len=0.2, verbose=False):
    for fcsv in os.listdir(folder):
        if not fcsv.endswith('.csv') or '_' in fcsv:
            continue
        print('\n=================================================================')
        print('Reading ' + os.path.join(folder, fcsv))
        df = pd.read_csv(os.path.join(folder, fcsv))

        resSeqs = df['resSeq'].drop_duplicates()
        L = int(int(len(resSeqs) / 4.0))
        FIXED_RID = [0, L, 2 * L]  # the third (last) rigid body is the greater one

        RESIDUES, natoms = crf.readResidues(df)

        # Determining which rigid body will be the first one to be realized and in which order (natural or backward).
        # The first candidate is always the greater rigid body.

        rb1size = FIXED_RID[1] - FIXED_RID[0] + 1
        rb2size = FIXED_RID[2] - FIXED_RID[1] + 1
        rb3size = len(resSeqs) - FIXED_RID[2] + 1
        greater_rb = np.argmax(np.array([rb1size, rb2size, rb3size]))

        isha0 = True if 'HA' in RESIDUES[FIXED_RID[0]] else False
        ish0p1 = True if 'H' in RESIDUES[FIXED_RID[0] + 1] else False
        ish1 = True if 'H' in RESIDUES[FIXED_RID[1]] else False
        isha1 = True if 'HA' in RESIDUES[FIXED_RID[1]] else False
        ish1p1 = True if 'H' in RESIDUES[FIXED_RID[1] + 1] else False
        ish2 = True if 'H' in RESIDUES[FIXED_RID[2]] else False
        isha2 = True if 'HA' in RESIDUES[FIXED_RID[2]] else False
        ish2p1 = True if 'H' in RESIDUES[FIXED_RID[2] + 1] else False

        # When the THIRD rigid body is the greater one...
        if greater_rb == 2:
            # the THIRD rigid body is the first one to be realized and it can just be done by using its backward order.
            if ish2p1 and isha2:
                ifixedha = 2
                isnatorder = False
            else:
                if rb2size > rb1size:
                    # Initiating the realization at the SECOND rigid body in the backward order
                    if ish1p1 and isha1:
                        ifixedha = 1
                        isnatorder = False
                    else:
                        # Initiating the realization at the SECOND rigid body in the natural order
                        if ish2 and isha2:
                            ifixedha = 2
                            isnatorder = True
                else:
                    # Initiating the realization at the FIRST rigid body in the backward order
                    if ish0p1 and isha0:
                        ifixedha = 0
                        isnatorder = False
                    else:
                        # Initiating the realization at the FIRST rigid body in the backward order
                        if ish1 and isha1:
                            ifixedha = 1
                            isnatorder = True
        else:
            # When the SECOND rigid body is the greater one...
            if greater_rb == 1:
                # Initiating the realization at the SECOND rigid body in the natural order
                if ish2 and isha2:
                    ifixedha = 2
                    isnatorder = True
                else:
                    # Initiating the realization at the SECOND rigid body in the backward order
                    if ish1p1 and isha1:
                        ifixedha = 1
                        isnatorder = False
                    else:
                        if rb1size > rb3size:
                            # Initiating the realization at the FIRST rigid body in the natural order
                            if ish1 and isha1:
                                ifixedha = 1
                                isnatorder = True
                            else:
                                # Initiating the realization at the FIRST rigid body in the backward order
                                if ish0p1 and isha0:
                                    ifixedha = 0
                                    isnatorder = False
                        else:
                            # Initiating the realization at the THIRD rigid body, which can just be done by using
                            # the backward order
                            if ish2p1 and isha2:
                                ifixedha = 2
                                isnatorder = False
            # When the FIRST rigid body is the greater one...
            else:
                # Initiating the realization at the FIRST rigid body in the natural order
                if ish1 and isha1:
                    ifixedha = 1
                    isnatorder = True
                else:
                    # Initiating the realization at the FIRST rigid body in the backward order
                    if ish0p1 and isha0:
                        ifixedha = 0
                        isnatorder = False
                    else:
                        if rb2size > rb3size:
                            # Initiating the realization at the SECOND rigid body in the natural order
                            if ish2 and isha2:
                                ifixedha = 2
                                isnatorder = True
                            else:
                                # Initiating the realization at the SECOND rigid body in the backward order
                                if ish1p1 and isha1:
                                    ifixedha = 1
                                    isnatorder = False
                        else:
                            # Initiating the realization at the THIRD rigid body, which can just be done by using
                            # the backward order
                            if ish2p1 and isha2:
                                ifixedha = 2
                                isnatorder = False

        # deleting hydrogen atoms info (some commands in the readResidues routine needs also to be uncommented).
        RESIDUES, natoms = crf.readResidues(df, resSeqs.iloc[FIXED_RID[ifixedha]], istoremoveH=True, isnatorder=isnatorder)

        print('fcsv ' + os.path.join(folder, fcsv))
        fdat = os.path.join(folder, fcsv).replace('.csv', '.dat')
        fid = open(fdat, 'w')
        print('fid: ', fdat)

        V = []
        XYZ = [None for _ in range(natoms)]
        for residue in RESIDUES:
            XYZ[residue['N']['nid']] = residue['N']['XYZ']
            XYZ[residue['CA']['nid']] = residue['CA']['XYZ']
            XYZ[residue['C']['nid']] = residue['C']['XYZ']
            V.append(residue['N'])
            V.append(residue['CA'])
            V.append(residue['C'])

            if 'H' in residue:
                XYZ[residue['H']['nid']] = residue['H']['XYZ']
                V.append(residue['H'])

            if 'HA' in residue:
                XYZ[residue['HA']['nid']] = residue['HA']['XYZ']
                V.append(residue['HA'])

        # anchors
        row = 'INFO: COLUMN RESSEQ ATOM  INDEX    X        Y         Z'
        if verbose: print(row)
        fid.write(row + '\n')
        for rid in FIXED_RID:
            residue = RESIDUES[rid]
            row = 'DATA: ANCHOR %6d   CA %6d' % (
                residue['resSeq'], residue['CA']['nid'] + 1) # soma-se 1 para que a indexacao comece em 1 e nao em 0?
            row = row + (' %8.5f %8.5f %8.5f' % residue['CA']['XYZ'])
            if verbose: print(row)
            fid.write(row + '\n')

        # INTERVAL CONSTRAINT
        rng = np.random.default_rng(seed=129)
        if 'HA' in RESIDUES[FIXED_RID[ifixedha]]:
            j = RESIDUES[FIXED_RID[ifixedha]]['HA']['nid']
        else:
            print('WARNING: The %d-th residue does not have "HA" atom to be used to calculate the interval distance.'
                  % RESIDUES[FIXED_RID[ifixedha]]['resSeq'])
            break

        if isnatorder:
            if 'H' in RESIDUES[FIXED_RID[ifixedha]]:
                i = RESIDUES[FIXED_RID[ifixedha]]['H']['nid']
            else:
                print('WARNING: The %d-th residue does not have "H" atom to be used to calculate the interval distance.'
                      % RESIDUES[FIXED_RID[ifixedha]]['resSeq'])
                break
        else:
            if 'H' in RESIDUES[FIXED_RID[ifixedha] + 1]:
                i = RESIDUES[FIXED_RID[ifixedha] + 1]['H']['nid']
            else:
                print('WARNING: The %d-th residue does not have "H" atom to be used to calculate the interval distance.'
                      % RESIDUES[FIXED_RID[ifixedha] + 1]['resSeq'])
                break

        vi = V[i]
        vj = V[j]

        # interval_len = 1.5
        print('interval_len:', interval_len)
        lij, uij = crf.generate_interval_distH(crf.distance(vi['XYZ'], vj['XYZ']), interval_len, rng)

        print('Exact interval distance: ', crf.distance(vi['XYZ'], vj['XYZ']))

        row = '\nINFO: COLUMN RES_I RES_J  ATM_I ATM_J IDX_I IDX_J         LIJ             UIJ'
        print(row)
        fid.write(row + '\n')
        row = 'DATA: INTERV %5d %5d' % (vi['resSeq'], vj['resSeq'])
        row += '%7s %5s' % (vi['name'], vj['name'])
        row += '%6d %5d' % (vi['nid'] + 1, vj['nid'] + 1)
        row += '%16.8f' % lij
        row += '%16.8f' % uij
        print(row)
        fid.write(row + '\n')
        itvAtomIdi = vi['nid']+1
        itvAtomIdj = vj['nid']+1

        # first rigid body
        R = [RESIDUES[0]['CA']['nid']]
        R.append(RESIDUES[0]['C']['nid'])
        for rid in range(FIXED_RID[0]+1, FIXED_RID[1]):
            R.append(RESIDUES[rid]['N']['nid'])
            R.append(RESIDUES[rid]['CA']['nid'])
            R.append(RESIDUES[rid]['C']['nid'])
        R.append(RESIDUES[FIXED_RID[1]]['N']['nid'])
        R.append(RESIDUES[FIXED_RID[1]]['CA']['nid'])
        crf.writeRIGID(fid, 0, R, V, XYZ)

        # hydrogen neighbourhood of FIRST rigid body backbone atoms
        H1 = []
        isthereHA123 = [False for i in range(len(FIXED_RID))] # Is there an alpha hydrogen in each fixed residue?
        if 'HA' in RESIDUES[0]:
            H1.append([RESIDUES[0]['HA']['nid'], RESIDUES[0]['CA']['nid'], RESIDUES[0]['C']['nid']])
            # Yes, there is an alpha hydrogen in the first fixed residue and it is in the first position of 'H1'.
            # It also will be in the last position of 'H3'.
            isthereHA123[0] = True
        for rid in range(FIXED_RID[0] + 1, FIXED_RID[1]):
            if 'H' in RESIDUES[rid]:
                H1.append([RESIDUES[rid]['H']['nid'], RESIDUES[rid - 1]['CA']['nid'],
                          RESIDUES[rid - 1]['C']['nid'], RESIDUES[rid]['N']['nid'], RESIDUES[rid]['CA']['nid']])
            if 'HA' in RESIDUES[rid]:
                H1.append([RESIDUES[rid]['HA']['nid'], RESIDUES[rid]['N']['nid'],
                          RESIDUES[rid]['CA']['nid'], RESIDUES[rid]['C']['nid']])
        if 'H' in RESIDUES[FIXED_RID[1]]:
            H1.append([RESIDUES[FIXED_RID[1]]['H']['nid'], RESIDUES[FIXED_RID[1] - 1]['CA']['nid'],
                      RESIDUES[FIXED_RID[1] - 1]['C']['nid'], RESIDUES[FIXED_RID[1]]['N']['nid'],
                      RESIDUES[FIXED_RID[1]]['CA']['nid']])
        if 'HA' in RESIDUES[FIXED_RID[1]]:
            H1.append([RESIDUES[FIXED_RID[1]]['HA']['nid'], RESIDUES[FIXED_RID[1]]['N']['nid'],
                      RESIDUES[FIXED_RID[1]]['CA']['nid']])
            # Yes, there is an alpha hydrogen in the second fixed residue and it is in the last position of 'H1'.
            # It also will be in the first position of 'H2'.
            isthereHA123[1] = True
        crf.writeHYDRO(fid, 0, H1, V, XYZ, itvAtomIdi, itvAtomIdj, interval_len, rng)

        # second rigid body
        R = [RESIDUES[FIXED_RID[1]]['CA']['nid']]
        R.append(RESIDUES[FIXED_RID[1]]['C']['nid'])
        for rid in range(FIXED_RID[1]+1, FIXED_RID[2]):
            R.append(RESIDUES[rid]['N']['nid'])
            R.append(RESIDUES[rid]['CA']['nid'])
            R.append(RESIDUES[rid]['C']['nid'])
        R.append(RESIDUES[FIXED_RID[2]]['N']['nid'])
        R.append(RESIDUES[FIXED_RID[2]]['CA']['nid'])
        crf.writeRIGID(fid, 1, R, V, XYZ)

        # hydrogen neighbourhood of SECOND rigid body backbone atoms
        H2 = []
        if 'HA' in RESIDUES[FIXED_RID[1]]:
            # Note: CA of this residues was captured in the FIRST rigid body
            H2.append([RESIDUES[FIXED_RID[1]]['HA']['nid'], RESIDUES[FIXED_RID[1]]['C']['nid']])
        for rid in range(FIXED_RID[1] + 1, FIXED_RID[2]):
            if 'H' in RESIDUES[rid]:
                H2.append([RESIDUES[rid]['H']['nid'], RESIDUES[rid - 1]['CA']['nid'],
                          RESIDUES[rid - 1]['C']['nid'], RESIDUES[rid]['N']['nid'], RESIDUES[rid]['CA']['nid']])
            if 'HA' in RESIDUES[rid]:
                H2.append([RESIDUES[rid]['HA']['nid'], RESIDUES[rid]['N']['nid'],
                          RESIDUES[rid]['CA']['nid'], RESIDUES[rid]['C']['nid']])
        if 'H' in RESIDUES[FIXED_RID[2]]:
            H2.append([RESIDUES[FIXED_RID[2]]['H']['nid'], RESIDUES[FIXED_RID[2] - 1]['CA']['nid'],
                      RESIDUES[FIXED_RID[2] - 1]['C']['nid'], RESIDUES[FIXED_RID[2]]['N']['nid'],
                      RESIDUES[FIXED_RID[2]]['CA']['nid']])
        if 'HA' in RESIDUES[FIXED_RID[2]]:
            H2.append([RESIDUES[FIXED_RID[2]]['HA']['nid'], RESIDUES[FIXED_RID[2]]['N']['nid'],
                      RESIDUES[FIXED_RID[2]]['CA']['nid']])
            # Yes, there is an alpha hydrogen in the third fixed residue and it is in the last position of 'H2'.
            # It also will be in the first position of the 'H3'.
            isthereHA123[2] = True
        crf.writeHYDRO(fid, 1, H2, V, XYZ, itvAtomIdi, itvAtomIdj, interval_len, rng)

        # third rigid body
        R = [RESIDUES[FIXED_RID[2]]['CA']['nid']]
        R.append(RESIDUES[FIXED_RID[2]]['C']['nid'])
        for rid in range(FIXED_RID[2]+1, len(RESIDUES)):
            R.append(RESIDUES[rid]['N']['nid'])
            R.append(RESIDUES[rid]['CA']['nid'])
            R.append(RESIDUES[rid]['C']['nid'])
        R.append(RESIDUES[0]['N']['nid'])
        R.append(RESIDUES[0]['CA']['nid'])
        crf. writeRIGID(fid, 2, R, V, XYZ)

        # hydrogen neighbourhood of THIRD rigid body backbone atoms
        H3 = []
        if 'HA' in RESIDUES[FIXED_RID[2]]:
            # Note: CA of this residues was captured in the SECOND rigid body
            H3.append([RESIDUES[FIXED_RID[2]]['HA']['nid'], RESIDUES[FIXED_RID[2]]['C']['nid']])
        for rid in range(FIXED_RID[2] + 1, len(RESIDUES)):
            if 'H' in RESIDUES[rid]:
                H3.append([RESIDUES[rid]['H']['nid'], RESIDUES[rid - 1]['CA']['nid'],
                          RESIDUES[rid - 1]['C']['nid'], RESIDUES[rid]['N']['nid'], RESIDUES[rid]['CA']['nid']])
            if 'HA' in RESIDUES[rid]:
                H3.append([RESIDUES[rid]['HA']['nid'], RESIDUES[rid]['N']['nid'],
                          RESIDUES[rid]['CA']['nid'], RESIDUES[rid]['C']['nid']])
        if 'HA' in RESIDUES[FIXED_RID[0]]:
            # Note: the CA of this residue was captured in the FIRST rigid body
            H3.append([RESIDUES[FIXED_RID[0]]['HA']['nid'], RESIDUES[FIXED_RID[0]]['N']['nid']])
        crf.writeHYDRO(fid, 2, H3, V, XYZ, itvAtomIdi, itvAtomIdj, interval_len, rng)


        # additional constraints N-C
        row = '\nINFO: COLUMN RES_I RES_J  ATM_I ATM_J IDX_I IDX_J   LIJ     UIJ'
        if verbose: print(row)
        fid.write(row + '\n')
        i = RESIDUES[0]['N']['nid']  # last N of the third rigid body
        j = RESIDUES[0]['C']['nid']  # first C of the first rigid body
        crf.writeLNK(fid, R, V, i, j, XYZ)
        # last N of the first rigid body
        i = RESIDUES[FIXED_RID[1]]['N']['nid']
        # first C of the second rigid body
        j = RESIDUES[FIXED_RID[1]]['C']['nid']
        crf.writeLNK(fid, R, V, i, j, XYZ)
        # last N of the second rigid body
        i = RESIDUES[FIXED_RID[2]]['N']['nid']
        # first C of the third rigid body
        j = RESIDUES[FIXED_RID[2]]['C']['nid']
        crf.writeLNK(fid, R, V, i, j, XYZ)

        fid.close()

        fsol = fdat.replace('.dat','.sol')
        print('\nWriting ' + fsol)
        with open(fsol, 'w') as fid:
            row = 'RESID    NAME      NID     X        Y        Z'
            if verbose: print(row)
            fid.write(row + '\n')
            for v in V:
                row = '%5d %7s %8d ' % (v['resSeq'], v['name'],  v['nid'] + 1)
                row += '%8.5f %8.5f %8.5f' % v['XYZ']
                if verbose: print(row)
                fid.write(row + '\n')


def main(interval_len=0.2):
    folders = ['DATA_LOOP_04', 'DATA_LOOP_08', 'DATA_LOOP_12']
    for folder in folders:
        print('Creating loops from folder ' + folder)
        create_hclp_instances(folder, interval_len=interval_len)


if __name__ == "__main__":
    # folder = 'DATA_LOOP_12'
    # if len(sys.argv) > 1:
    #     folder = sys.argv[1]

    # print('Creating loops from folder ' + folder)
    # create_hclp_instances(folder)

    this_len = 1.0
    if len(sys.argv) > 1:
        this_len = float(sys.argv[1])
    main(interval_len=this_len)