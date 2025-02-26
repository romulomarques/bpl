import pandas as pd
import os
import sys
import numpy as np
import create_instances_functions as crf

def create_hclp_instances(folder, verbose=False):
    for fcsv in os.listdir(folder):
        if not fcsv.endswith('.csv') or '_' in fcsv:
            continue
        print('\n=================================================================')
        print('Reading ' + os.path.join(folder, fcsv))
        df = pd.read_csv(os.path.join(folder, fcsv))

        resSeqs = df['resSeq'].drop_duplicates()
        L = int(int(len(resSeqs) / 4.0))
        FIXED_RID = [0, L, 2 * L]  # the third (last) rigid body is the greater one

        RESIDUES, natoms = crf.readResidues(df, istoremoveAllH=True)

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
        i = RESIDUES[0]['N']['nid']
        j = RESIDUES[FIXED_RID[1]]['CA']['nid']
        a = RESIDUES[FIXED_RID[0]]['CA']['nid']
        b = RESIDUES[FIXED_RID[2]]['CA']['nid']
        lij, uij = crf.distance_range(RESIDUES[0]['N']['nid'], a, b, RESIDUES[FIXED_RID[1]]['CA']['nid'], V)

        vi = V[i]
        vj = V[j]

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


def main():
    folders = ['DATA_LOOP_04', 'DATA_LOOP_08', 'DATA_LOOP_12']
    for folder in folders:
        print('Creating loops from folder ' + folder)
        create_hclp_instances(folder)


if __name__ == "__main__":
    # folder = 'DATA_LOOP_12'
    # if len(sys.argv) > 1:
    #     folder = sys.argv[1]

    # print('Creating loops from folder ' + folder)
    # create_hclp_instances(folder)

    main()