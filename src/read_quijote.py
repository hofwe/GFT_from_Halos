import readfof

snapdir = '../samples'
snapnum = 4
redshift = 0.0

FoF = readfof.FoF_catalog(snapdir, snapnum, long_ids=False, swap=False, SFR=False, read_IDs=False)

pos = FoF.GroupPos/1e3
mass = FoF.GroupMass*1e10
