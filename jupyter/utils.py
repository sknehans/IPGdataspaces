import os
import unicodedata
from glob import glob
import datetime

import pandas as pd
import numpy as np

from rapidfuzz import fuzz

def prepare_dimensions_data(filenames, columns_to_keep, places=None, year = datetime.date.today().year):
    
    # Load all the Excels as downloaded from Dimensions
    df = [ None for _ in range(len(filenames)) ]
    for i in range(len(df)):
        df[i] = pd.read_excel(filenames[i], header=1)
        print(filenames[i])

    # Merge all excels into one
    df = pd.concat(df, ignore_index=True)

    # Discard seminars and preprints, except those prepints that have been published in the last year
    # This should prevent double-counting papers
    df = df[(df['Publication Type'] != 'Preprint') | ( (df['Publication Type'] == 'Preprint') & (df['PubYear'] >= year-1 ))]
    df = df[df['Publication Type'] != 'Seminar']

    # Lowercase the paper keywords
    df['MeSH terms'] = df['MeSH terms'].str.lower()

    # Re-index
    df.index = range(len(df))

    # Force all author names to be ascii
    for i in range(len(df)):
        for column in ['Authors', 'Authors (Raw Affiliation)', 'Corresponding Authors']:
            if not pd.isna(df.loc[i,column]):
                text = unicodedata.normalize('NFD', df.loc[i,column])
                text = text.encode('ascii', 'ignore').decode("utf-8")
                df.loc[i,column] = text.replace('-','').replace('.','')
    
    # Drop papers where no author is geographically associated to any of the indicated places
    # (Only if places are specified)
    if places is not None:
        isplace = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            isplace[i] = any([place in df.loc[i, 'Authors (Raw Affiliation)'].lower() for place in places])
        df = df.iloc[isplace]
        
    # Only keep a few columns
    # Drop any papers that might be duplicated
    df = df.loc[:, columns_to_keep].drop_duplicates(subset=[columns_to_keep[0]])
    
    return df



# `plantssns` is the list of ISSNs corresponding to plant-specific journals
# Get the papers published in plant journals
def mask_plant_journals(df, plantssns):
    isplantjournal = np.zeros(len(df), dtype=bool)
    for i,idx in enumerate(df.index):
        if not pd.isna(df.loc[idx, 'ISSN']):
            issn = df.loc[idx, 'ISSN'].split(', ')
            isplantjournal[i] = any([ issn[j] in plantssns for j in range(len(issn)) ])
    
    return isplantjournal
    


# Get the papers that were categorized as Plant Biology or Horticulture
def mask_plant_anzsrc(df, ANZSRC = ['3108 Plant Biology','3008 Horticultural']):
    isplantanz = np.zeros(len(df), dtype=bool)
    for i,idx in enumerate(df.index):
        if not pd.isna(df.loc[idx, 'Fields of Research (ANZSRC 2020)']):
            anzcodes = df.loc[idx,'Fields of Research (ANZSRC 2020)']
            isplantanz[i] = any([ anz in anzcodes for anz in ANZSRC ])
    
    return isplantanz



# Count how many plant-related keywords are in the paper
def count_plant_mesh(df, meshterms):
    isplantmesh = np.zeros(len(df), dtype=int)
    for i,idx in enumerate(df.index):
        mscores = np.zeros(len(meshterms), dtype=int)
        if not pd.isna(df.loc[idx, 'MeSH terms']):
            terms = df.loc[idx, 'MeSH terms']
            for j in range(len(mscores)):
                mscores[j] = terms.count(meshterms[j])
            isplantmesh[i] = np.sum(mscores)
    return isplantmesh



# Get only the corresponding authors that are affiliated to the institution(s) we are focused on
# (Optional) but discard those with a dual affiliation to another institution(s)
# And the number of papers associated to each
# Also get the index number these papers correspond to
def corresponding_authors_from_institute(df, institutes, exclude_list=None):
    
    instlist = [inst.replace('.','').replace('-','') for inst in institutes]
    if exclude_list is not None:
        exclude_list = [inst.replace('.','').replace('-','') for inst in exclude_list]
    corrs = []
    idx = []
    
    if exclude_list is None:
        for i in range(len(df)):
            names = (df.iloc[i]['Corresponding Authors']).split('); ')
            for name in names:
                if any( [institute in name for institute in instlist] ):
                    corrs.append(name.split(' (')[0])
                    idx.append(df.iloc[i].name)
    else:
        for i in range(len(df)):
            names = (df.iloc[i]['Corresponding Authors']).split('); ')
            for name in names:
                if any( [institute in name for institute in instlist] ) and not any([institute in name for institute in exclude_list]):
                    corrs.append(name.split(' (')[0])
                    idx.append(df.iloc[i].name)


    uq, cts = np.unique(corrs, return_counts=True)
    cts = pd.Series(cts, index=uq)
    
    return cts, idx



# Remove corresponding authors with long names because they are not really people but parsing mistakes
def remove_long_corresponding(cts, alpha=0.15, iqr_range=1.5):
    
    uq = cts.index
    uqlens = np.array(list(map(len,uq)))
    q1,q3 = np.quantile(uqlens, [alpha, 1-alpha])
    uqmask = uqlens < q3 + iqr_range*(q3 - q1)
    print('Dropped:\n', uq[~uqmask], '\n--',sep='')
    uq = uq[uqmask]
    cts = cts[uqmask]
    cts = pd.Series(cts, index=uq)
    
    return cts


def add_blanks(s):
    word = ''
    t = s.title()
    for i in range(len(t)):
        if t[i] == s[i]:
            word += t[i]
        else:
            word += ' ' + t[i].upper()
    return word

# Compute a square matrix of distances: how similar is one name to another?
# This matrix will later help us match different names that refer to the same author
# Last name and first name are treated separately: then we take the minimum between the two
def fuzzy_matching(name1, name2, accept_value = 98):
    
    lname1, fname1 = name1.split(', ')
    inits1 = [x[0] for x in fname1.split(' ')]
    
    lname2, fname2 = name2.split(', ')
    inits2 = [x[0] for x in fname2.split(' ')]
    
    #If none of the initials match, then assume that names are not equal and move on
    if not any([x in inits1 for x in inits2]):
        return -1, -1
    
    # Sometimes, different names are just different casings
    # e.g. Zhong, GanYuan vs Zhong, Ganyuan
    fuzzscore = fuzz.partial_ratio(name1.casefold(), name2.casefold())
    if fuzzscore >= accept_value:
        return fuzzscore, fuzzscore
    
    # Add blank spaces for first names reduced to intials
    # e.g. Riedell, WE --> Riedell, W E
                    
    fname1 = add_blanks(fname1)
    inits1 = [x[0] for x in fname1.split(' ')]
    
    fname2 = add_blanks(fname2)
    inits2 = [x[0] for x in fname2.split(' ')]
    
    # If the number of initials is larger in the initial name, then remove the blanks from the first name and titlecase:
    # e.g. Wang, Hongshu vs Wang, Hong S --> Wang, Hongshu vs Wang, Hongs
    #
    # However, you don't want to join two separate initials
    # e.g. You want to keep 'Babiker, E M', not make it 'Babiker, EM'
    if ( len(fname2) >= 2*len(inits2) ) & ( len(inits1) < len(inits2) ):
        fname2 = fname2.replace(' ', '').title()
        inits2 = [x[0] for x in fname2.split(' ')]
    
    # If the first name is just intials, then make the original first name initials as well
    # e.g. Hibbard, Bruce vs Hibbard, B E --> Hibbard, B vs Hibbard, B E
    if ' '.join(inits2) == fname2:
        fname1 = ' '.join(inits1)
    
    ffz = max([fuzz.token_set_ratio(fname1, fname2) , fuzz.partial_ratio(fname1, fname2)])
    
    # compare the last names, which we expect to be the same even with different naming conventions
    lfz = max([fuzz.token_set_ratio(lname1, lname2) , fuzz.partial_ratio(lname1, lname2)])
    
    return ffz, lfz
    
    
# Compute a square matrix of distances: how similar is one name to another?
# This matrix will later help us match different names that refer to the same author
# Last name and first name are treated separately: then we take the minimum between the two
def fuzzy_matrix(pnum):
    
    foo = pnum.to_frame('Pubs')
    foo['N'] = range(len(foo))
    foo['len'] = list(map(len, pnum.index))
    foo['name'] = foo.index.str.casefold()
    foo = foo.sort_values(by=['len','Pubs','name'], ascending=[False, False, True])
    
    firstfz = pd.DataFrame(-1, index=foo.index, columns=foo.index, dtype=float)
    lastfz = pd.DataFrame(-1, index=foo.index, columns=foo.index, dtype=float)
    
    
    for i in range(len(firstfz)-1):
        name1 = firstfz.index[i]
        for j in range(i+1, len(firstfz)):
            name2 = firstfz.index[j]
            
            firstfz.iloc[i,j], lastfz.iloc[i,j] = fuzzy_matching(name1, name2)
            firstfz.iloc[j,i], lastfz.iloc[j,i] = firstfz.iloc[i,j], lastfz.iloc[i,j]
            
    # The fuzzy scores is the minimum between the last and first name matching
    fz = firstfz.where(firstfz < lastfz, lastfz)

    return fz
    

# Get a list with only one true copy of every author along their total publication count
# We keep the longest name as the true copy
def fuzzymatching_authors(pnum, fz, tol=90):
        
    uqdict = dict()
    uqset = set(fz.index)
    ctset = pnum.copy()

    print('Started with:\t', len(uqset), '\n')
    for i in range(len(fz)-1):
        name = fz.index[i]
        foo = fz.iloc[ i: , i+1:].loc[name, fz.loc[name] >= tol]
        if len(foo) > 0:
            # If there are other names that are fuzzy-matched
            # Remove those copies from the author list
            # Add their papers to this true name
            
            uqset = uqset - set(foo.index)
            ctset[name] += pnum[foo.index].values.sum()
            uqdict[name] = foo.index.values.tolist()

    uqset = sorted(list(uqset))
    ctset = ctset.loc[uqset]
    ctdict = dict()
    
    for name in ctset.index:
        ctdict[name] = [name]
        if name in uqdict:
            ctdict[name] += uqdict[name]
            print(name, '-->', uqdict[name], sep='\t')
            
        # Add a initial-less variant if it doesn't exist
        lname, fname = name.split(', ')
        fnames = fname.split(' ')
        for i in range(1, len(fnames)):
            alt = lname + ', ' + fnames[0] + ' ' + fnames[i][0]
            if alt not in ctdict[name]:
                ctdict[name].append(alt)
        
        alt = lname + ', ' + fnames[0]
        if alt not in ctdict[name]:
            ctdict[name].append(alt)
    
    print('\nAfter matching:\t', len(uqset), sep='')
    
    return ctset, ctdict
