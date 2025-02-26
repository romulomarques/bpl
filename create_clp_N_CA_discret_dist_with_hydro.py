import pandas as pd
import os
import sys
import numpy as np
import create_instances_functions as crf

def create_hclp_instances(folder, interval_len=0.2, verbose=False):
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

        print('fcsv ' + os.path.join(folder, fcsv))
        fdat = os.path.join(folder, fcsv).replace('.csv','.dat')
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


        # Using the discretization interval distance from Labiak-Lavor-Souza order.
        i = RESIDUES[0]['N']['nid']
        j = RESIDUES[FIXED_RID[1]]['CA']['nid']
        a = RESIDUES[FIXED_RID[0]]['CA']['nid']
        b = RESIDUES[FIXED_RID[2]]['CA']['nid']
        lij, uij = crf.distance_range(RESIDUES[0]['N']['nid'], a, b, RESIDUES[FIXED_RID[1]]['CA']['nid'], V)

        vi = V[i]
        vj = V[j]

        # interval_len = 1.5
        print('interval_len:', interval_len)

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
        itvAtomIdi = -1 # In this case, the first atom of the 'INTERV' distance is not a hydrogen
        itvAtomIdj = -1 # In this case, the second atom of the 'INTERV' distance is not a hydrogen
        rng = np.random.default_rng(seed=129)

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


        # hydrogen neighbourhood between atoms from the FIRST and SECOND rigid bodies
        H = [H1, H2, H3]
        if isthereHA123[1]:
            posHAa_b = len(H[0]) - 1
            posHAb_a = 0
        else:
            posHAa_b = len(H[0])
            posHAb_a = len(H[1])
        crf.writeCROSSHYDRO(fid, 0, 1, H[0], H[1], posHAa_b, posHAb_a, V, XYZ, itvAtomIdi, itvAtomIdj,
                    interval_len, rng)

        # hydrogen neighbourhood between atoms from the SECOND and THIRD rigid bodies
        if isthereHA123[2]:
            posHAa_b = len(H[1]) - 1
            posHAb_a = 0
        else:
            posHAa_b = len(H[1])
            posHAb_a = len(H[2])
        crf.writeCROSSHYDRO(fid, 1, 2, H[1], H[2], posHAa_b, posHAb_a, V, XYZ, itvAtomIdi, itvAtomIdj,
                        interval_len, rng)

        # hydrogen neighbourhood between atoms from the THIRD and FIRST rigid bodies
        if isthereHA123[0]:
            posHAa_b = len(H[2]) - 1
            posHAb_a = 0
        else:
            posHAa_b = len(H[2])
            posHAb_a = len(H[0])
        crf.writeCROSSHYDRO(fid, 0, 2, H[0], H[2], posHAb_a, posHAa_b, V, XYZ, itvAtomIdi, itvAtomIdj,
                        interval_len, rng)


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

    this_len = 0.2
    if len(sys.argv) > 1:
        this_len = float(sys.argv[1])
    main(interval_len=this_len)