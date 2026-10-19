"""Equal-weight calibration-seed groups, expanded back to actual raw columns."""
import re
import numpy as np

def seed_expansion(ids, weights, round_name, expected=3):
    groups={}
    for i,label in enumerate(ids):
        rnd,atom,case=label.split('/')
        key=label
        if rnd==round_name:
            match=re.fullmatch(r'(.+)-seed(\d+)',atom)
            if match is None:raise ValueError('Seed-group atom lacks explicit seed suffix: '+label)
            key=rnd+'/'+match[1]+'/'+case
        groups.setdefault(key,[]).append(i)
    if not any(k.startswith(round_name+'/') for k in groups):
        raise ValueError('Requested seed-group round absent from support')
    expansion=np.zeros((len(ids),len(groups)))
    aggregate=[];description=[]
    for j,(key,members) in enumerate(groups.items()):
        if key.startswith(round_name+'/') and len(members)!=expected:
            raise ValueError('Incomplete seed group: '+key)
        values=np.asarray(weights)[members]
        if not np.allclose(values,values[0],rtol=0,atol=1e-15):
            raise ValueError('Initial weights violate equal-seed constraint: '+key)
        expansion[members,j]=1/len(members);aggregate.append(values.sum())
        if len(members)>1:description.append({'group':key,'members':[ids[i] for i in members]})
    return expansion,np.asarray(aggregate),description
