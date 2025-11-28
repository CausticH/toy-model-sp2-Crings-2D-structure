Setting the generation configurations in section 1 of the code multiple_symmetric_growth_db.py

To start generate and write log in log.txt
python multiple_symmetric_growth_db.py > log.txt 2>&1 

Use export_id_xyz.py to export .xyz file for molecules of different id(change start and end in the code)

Use plot_ring_num.py to check the number of total rings of mols by a bar graph

Use stats_multitype.py to export different statistic data(switch true/false in the code)

Use extract_xyz_sym.py to draw .xyz file for molecules od different symmetry evenly


Tips: some useful commands for db (use ase db help to learn more)
ase db mol_0D.db sym=C2 --count
ase db mol_0D.db "nN>0" --count
ase db mol_0D.db -c id,formula,sym,defects,n4,n5,n6,n7,n8,nN,nB,nO,nS,hash24,task_id --limit 20