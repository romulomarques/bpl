import os
import sys
import numpy as np
import pandas as pd

def read_log(flog, loops, anchors, interval):
    print('Reading ' + flog)

    df = {'pdb': [], 'nnodes': [], 'nbbnodes': [], 'nedges': [], 'num': [], 'idist': [], 'nsols': [], 'nbbsols': [], 'tsecs': [],
          'min_eij': [], 'avg_eij': [], 'max_eij': [], 'min_rmsd': [], 'avg_rmsd': [], 'max_rmsd': []}

    loop_size = int(flog.split('DATA_LOOP_')[1].split('/')[0])
    pdb = flog.replace('_bb.log', '').split('/')[-1].split('_')[0]
    loop = loops[pdb + '_' + str(loop_size)]
    pdb += loop['chainID'] + '(' + f"{loop['frstRes']},{anchors[1]},{anchors[2]}" + ')'
    # pdb += loop['chainID'] + '_' + loop['frstRes']
    # pdb += ' (%s,%s)' % (anchors[1],anchors[2])
    df['pdb'] = '{' + pdb + '}'
    print(pdb)
    df['idist'] = '{[%.2f, %.2f]}' % (interval[0], interval[1])
    eij = []
    rmsd = []
    countBBatoms = 0
    with open(flog, 'r') as fid:
        for k, row in enumerate(fid.readlines()):
            row = row.replace('\n','')

            if 'atoms' in row:
                this_atoms = row.split('[')[1].split(']')[0].split()
                for atomsymbol in this_atoms:
                    countBBatoms = countBBatoms + 1 if atomsymbol != 'H' and atomsymbol != 'HA' else countBBatoms
            elif 'eijMax' in row:
                row = row.split()                
                eij.append(float(row[1].split('=')[1]))
                rmsd.append(float(row[2].split('=')[1]))
            elif 'num .....' in row:
                df['num'].append(int(row.split()[-1]))
            elif 'nnodes' in row:
                df['nnodes'].append(int(row.split()[-1]))
                df['nbbnodes'].append(countBBatoms)
            elif 'nedges' in row:   
                df['nedges'].append(int(row.split()[-1]))
            elif 'nsols ...' in row:
                df['nsols'].append(int(row.split()[-1]))
            elif 'nbbsols' in row:
                df['nbbsols'].append(int(row.split()[-1]))
            elif 'tsecs ...' in row:
                df['tsecs'].append(float(row.split()[-1]))
                df['min_eij'].append(np.min(eij) if len(eij) > 0 else None)
                df['avg_eij'].append(np.mean(eij) if len(eij) > 0 else None)
                df['max_eij'].append(np.max(eij) if len(eij) > 0 else None)
                df['min_rmsd'].append(np.min(rmsd) if len(rmsd) > 0 else None)
                df['avg_rmsd'].append(np.mean(rmsd) if len(rmsd) > 0 else None)
                df['max_rmsd'].append(np.max(rmsd) if len(rmsd) > 0 else None)
    
    for col in df:
        print('%12s ... %d' % (col, len(df[col])))
    
    # fxlsx = flog.replace('.log', '.xlsx')
    # fcsv = flog.replace('.log', '.csv')
    df = pd.DataFrame(df)
    print(df)
    # print('Writing ' + fxlsx)
    # df.to_excel(fxlsx, index=False)
    # print('Writing ' + fcsv)
    # df.to_csv(fcsv, index=False)
    return df


if __name__ == "__main__":
    loops = {}
    with open('table_loopDefs.csv', 'r') as fid:
        for k, row in enumerate(fid.readlines()):
            if k == 0: # skip header
                continue
            row = row.split(',')
            loops[row[0] + '_' + row[1]] = {'chainID':row[2], 'frstRes':row[3]}

    isthereH = False
    df = None
    # for wdir in ['DATA_LOOP_04','DATA_LOOP_08','DATA_LOOP_12']:
    for wdir in ['tests_idist_1dot0/results_essential_hydro_hl_order/DATA_LOOP_04','tests_idist_1dot0/results_essential_hydro_hl_order/DATA_LOOP_08','tests_idist_1dot0/results_essential_hydro_hl_order/DATA_LOOP_12']:
        for flog in os.listdir(wdir):
            if not flog.endswith('bb.log'):
                continue
            # read anchors, interval
            fdat = os.path.join(wdir, flog.split('_')[0] + '.dat')
            with open(fdat, 'r') as fid:
                anchors = []
                interval = []
                for row in fid.readlines():
                    if 'ANCHOR' in row:
                        anchors.append(row.split()[2])
                    elif 'INTERV' in row:
                        interval.append(float(row.split()[-2]))
                        interval.append(float(row.split()[-1]))
                    #     if row.split()[-5] == 'H' or row.split()[-5] == 'HA' or row.split()[-6] == 'H' or row.split()[-6] == 'HA':
                    #         isthereH = True
                    # elif 'RIGID' in row:
                    #     if row.split()[-5] == 'H' or row.split()[-5] == 'HA' or row.split()[-6] == 'H' or row.split()[-6] == 'HA':
                    #         isthereH = True

            
            flog = os.path.join(wdir, flog)

            if df is None:
                df = read_log(flog, loops, anchors, interval)
            else:
                df = pd.concat([df, read_log(flog, loops, anchors, interval)])
    print('Saving ' + 'results.xlsx')
    df = df.sort_values(by=['nbbnodes','pdb','num'])
    df = df.drop(['nbbnodes'], axis=1)
    df.to_csv('results.csv', index=False)
    df.to_excel('results.xlsx', index=False)