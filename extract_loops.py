import os
import numpy as np
import pandas as pd
from prody import measure as prodym
from read_pdb import read_pdb
from bpl import solveEQ3

os.mkdir('DATA_LOOP_04')
os.mkdir('DATA_LOOP_08')
os.mkdir('DATA_LOOP_12')

loopDefs = pd.read_csv('table_loopDefs.csv')

PDB = {}
ATOMS = ['C', 'CA', 'N', 'H', 'H1', 'H2', 'HA', 'HA1']

rng = np.random.default_rng(seed=149)

def calculateHA(coordsN, coordsCA, coordsC, coordsCB):
    vCAtoN = coordsN - coordsCA
    vCAtoN = vCAtoN / np.linalg.norm(vCAtoN)
    vCAtoC = coordsC - coordsCA
    vCAtoC = vCAtoC / np.linalg.norm(vCAtoC)
    vCAtoCB = coordsCB - coordsCA
    vCAtoCB = vCAtoCB / np.linalg.norm(vCAtoCB)

    # bond length of the HAi-CAi covalent bond (extracted from CRC Handbook of Chemistry and Physics - 95th ed.
    # - Csp3-H, case X_{3}-C-H)
    mean_dhaca = 1.099
    std_dhaca = 0.007
    lower_dhaca = mean_dhaca - std_dhaca
    upper_dhaca = mean_dhaca + std_dhaca
    dhaca = rng.uniform(lower_dhaca, upper_dhaca)

    #vCAtoHA = vCAtoN + vCAtoC
    #vCAtoHA = vCAtoHA / np.linalg.norm(vCAtoHA)
    #vCAtoHA = vCAtoHA + vCAtoCB
    vCAtoHA = vCAtoN + vCAtoC + vCAtoCB
    vCAtoHA = vCAtoHA / np.linalg.norm(vCAtoHA)

    return coordsCA - dhaca * vCAtoHA


def calculateH(coordsN, coordsCA, coordsCim1):
    # distances from Ni to C_{i-1} and CAi
    dncim1 = prodym.measure.getDistance(coordsN, coordsCim1)
    dnca = prodym.measure.getDistance(coordsN, coordsCA)

    # C_{i-1}-Ni-CAi bond angle in degree
    c_n_ca_angle = prodym.measure.getAngle(coordsCim1, coordsN, coordsCA, )
    # C_{i-1}-Ni-Hi bond angle in degree
    c_n_h_angle = ((360 - c_n_ca_angle) / 2) + 1
    # Hi-Ni-CAi bond angle in degree
    h_n_ca_angle = c_n_h_angle - 2

    # bond length of the Hi-Ni covalent bond (extracted from CRC Handbook of Chemistry and Physics - 95th ed. -
    # H-N(3) - case X_{3}-N-H)
    mean_dhn = 1.009
    std_dhn = 0.019
    lower_dhn = mean_dhn - std_dhn
    upper_dhn = mean_dhn + std_dhn
    dhn = rng.uniform(lower_dhn, upper_dhn)

    # distances from Hi to C_{i-1} and CAi calculated by the law of cosines.
    dhcim1 = np.sqrt(dncim1 ** 2 + dhn ** 2 - 2 * dncim1 * dhn * np.cos(np.deg2rad(c_n_h_angle)))
    dhca = np.sqrt(dhn ** 2 + dnca ** 2 - 2 * dhn * dnca * np.cos(np.deg2rad(h_n_ca_angle)))

    solved, xpos, xneg = solveEQ3(coordsCim1, coordsN, coordsCA, dhcim1, dhn, dhca)

    return solved, xpos, xneg


# reading pdb data
for fpdb in os.listdir('pdb'):
    if not fpdb.endswith('.pdb.gz'):
        continue
    pdbCode = fpdb.split('.')[0]
    fpdb = os.path.join('pdb', fpdb)
    pdbDataFrame = read_pdb(fpdb)
    PDB[pdbCode] = pdbDataFrame


# creating instances
for i, loop in loopDefs.iterrows():
    if loop.pdbCode not in PDB:
        print('PDB not found (%s)' % loop.pdbCode)
        continue
    print('\n' + loop.pdbCode + ':\n')
    df = PDB[loop.pdbCode]
    df = df[loop.resSeqFirst <= df['resSeq']]
    df = df[loop.resSeqLast  >= df['resSeq']]
    df = df.query("chainID=='%s'" % loop.chainID)

    # Note: the data of the second amino hydrogen of a residue (H2), if it exists, is not captured
    df = df.query("name=='C' | name=='CA' | name=='N' | name=='H' | name=='H1' | name=='HA' | name=='HA2' | name =='CB'"
                  " | name =='HA3'")
    df = df.query("altLoc==' ' | altLoc=='A'")

    # removing the amino hydrogen data of the first residue
    dfH = df.loc[(df['resSeq'] == df['resSeq'].iloc[0]) & (df['name'] == 'H')]
    #print(df.loc[(df['name'] == 'H') | (df['name'] == 'H1') | (df['name'] == 'HA') | (df['name'] == 'HA2')])
    if not dfH.empty:
        df.drop(dfH.index, inplace=True)
        print('The "H" of the first residue of %s was removed.' % loop.pdbCode)
    else:
        print('There is no "H" in the first residue of %s.' % loop.pdbCode)

    dfH1 = df.loc[(df['resSeq'] == df['resSeq'].iloc[0]) & (df['name'] == 'H1')]
    if not dfH1.empty:
        df.drop(dfH1.index, inplace=True)
        print('The "H1" of the first residue of %s was removed.' % loop.pdbCode)
    else:
        print('There is no "H1" in the first residue of %s.' % loop.pdbCode)
    print('\n')


    # storing, to each residue, the coords and the 'df' index of each atom.
    RESIDUES = {}
    idf = 0
    for j, atom in df.iterrows():
        if atom.resSeq not in RESIDUES:
            x = np.array([float(atom.x), float(atom.y), float(atom.z)], dtype=float)
            RESIDUES[atom.resSeq] = {atom['name']: {'x': x, 'idf': idf, 'resname': atom['resName']}}
        else:
            x = np.array([float(atom.x), float(atom.y), float(atom.z)], dtype=float)
            RESIDUES[atom.resSeq][atom['name']] = {'x': x, 'idf': idf, 'resname': atom['resName']}
        idf += 1

    RESIDUES = [{residue: RESIDUES[residue]} for residue in RESIDUES]


    H = []
    indexH = []
    HA = []
    indexHA = []
    isthereCB = False

    # calculating the coords of HA of the FIRST residue.
    for jres in RESIDUES[0]:
        coordsN = RESIDUES[0][jres]['N']['x']
        coordsCA = RESIDUES[0][jres]['CA']['x']
        coordsC = RESIDUES[0][jres]['C']['x']
        if 'HA' not in RESIDUES[0][jres] and 'HA2' not in RESIDUES[0][jres] and 'CB' in RESIDUES[0][jres]:
                isthereCB = True
                coordsCB = RESIDUES[0][jres]['CB']['x']

    if isthereCB:
        coordsHA = calculateHA(coordsN, coordsCA, coordsC, coordsCB)

        HA.append(coordsHA)
        indexHA.append(len(HA) - 1)

        dhah = np.linalg.norm(coordsHA - coordsHA)
    else:
        indexHA.append(-1)

    # the amino hydrogen atom of the FIRST residue are not needed
    indexH.append(-1)


    # calculating, to each residue (except the first one), the coords of each H and HA.
    for j in range(1, len(RESIDUES)):
        # RESIDUES[i] has just one pair key:value
        isthereCB = False
        isthereH = True
        for jres in RESIDUES[j]:
            coordsN = RESIDUES[j][jres]['N']['x']
            coordsCA = RESIDUES[j][jres]['CA']['x']
            coordsC = RESIDUES[j][jres]['C']['x']
            if 'HA' not in RESIDUES[j][jres] and 'HA2' not in RESIDUES[j][jres] and 'CB' in RESIDUES[j][jres]:
                isthereCB = True
                coordsCB = RESIDUES[j][jres]['CB']['x']

            for kres in RESIDUES[j-1]:
                if 'H' not in RESIDUES[j][jres] and 'H1' not in RESIDUES[j][jres] and \
                        RESIDUES[j][jres]['N']['resname'] != 'PRO':
                    isthereH = False
                    coordsCjm1 = RESIDUES[j-1][kres]['C']['x']

        if not isthereH:
            solved, xpos, xneg = calculateH(coordsN, coordsCA, coordsCjm1)

            if solved:
                H.append(xpos)
                indexH.append(len(H)-1)
            else:
                indexH.append(-1)
                print('Not solved!')
        else:
            print(loop.pdbCode + ':', 'possui H do residuo %d.' % j)
            indexH.append(-1)

        if isthereCB:
            coordsHA = calculateHA(coordsN, coordsCA, coordsC, coordsCB)

            HA.append(coordsHA)
            indexHA.append(len(HA)-1)
        else:
            print(loop.pdbCode + ':', f'Either the instance already has HA from residue {j}, or it has neither HA nor CB.')
            indexHA.append(-1)

    # inserting the new hydrogens in the dataframe of atoms (df) at the correct position according the PDB standard.
    counthydros = 0
    for j in range(0, len(RESIDUES)):
        count = 0
        if indexH[j] > -1:
            for jres in RESIDUES[j]:
                idfC = int(RESIDUES[j][jres]['C']['idf'])
            carbonline = df.iloc[(idfC + counthydros)]
            hydroline = pd.DataFrame({'RECORD': 'ATOM', 'serial': int(carbonline['serial']) + 2, 'name': 'H',
                                      'altLoc': ' ', 'resName': carbonline['resName'], 'chainID': carbonline['chainID'],
                                      'resSeq': carbonline['resSeq'], 'iCode': ' ', 'x': H[indexH[j]][0], 'y': H[indexH[j]][1],
                                      'z': H[indexH[j]][2], 'occupancy': 1.0, 'tempFactor': ' ', 'element': 'H', 'charge': ' '},
                                     index=[0])
            df = pd.concat([df.iloc[:(idfC + counthydros + 1)], hydroline, df.iloc[(idfC + counthydros + 1):]])
            count += 1
            print('Added H_%d' % j)
        else:
            print('Did NOT add H_%d' % j)

        if indexHA[j] > -1:
            for jres in RESIDUES[j]:
                idfC = int(RESIDUES[j][jres]['C']['idf'])
            carbonline = df.iloc[(idfC + counthydros)]
            hydroline = pd.DataFrame({'RECORD': 'ATOM', 'serial': int(carbonline['serial']) + 3, 'name': 'HA',
                                      'altLoc': ' ', 'resName': carbonline['resName'], 'chainID': carbonline['chainID'],
                                      'resSeq': carbonline['resSeq'], 'iCode': ' ', 'x': HA[indexHA[j]][0], 'y': HA[indexHA[j]][1],
                                      'z': HA[indexHA[j]][2], 'occupancy': 1.0, 'tempFactor': ' ', 'element': 'H', 'charge': ' '},
                                     index=[0])
            df = pd.concat([df.iloc[:(idfC + counthydros + count + 1)], hydroline, df.iloc[(idfC + counthydros + count + 1):]],
                           ignore_index=True)
            count += 1
            print('Added HA_%d' % j)
        else:
            print('Did NOT add HA_%d' % j)

        counthydros += count

    # removing from the dataframe of atoms (df) the beta carbon (CB) atom of non-glycine residues and
    # the second hydrogen atom bonded to the alpha carbon atom of glycine residues.
    df = df.query("name=='C' | name=='CA' | name=='N' | name=='H' | name=='H1' | name=='HA' | name=='HA2'")

    wdir = 'DATA_LOOP_%02d' % loop.loopSize
    if not os.path.isdir(wdir):
        os.mkdir(wdir)
    fcsv = os.path.join(wdir, loop.pdbCode + '.csv')
    df.to_csv(fcsv, index=False)
    
