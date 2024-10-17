nohup nextflow run /czlab/xuwen/scripts/Pre-fasta2ubam/fasta2ubam.nf \
    -profile hpc_slurm \
    -work-dir /czlab/xuwen/results/NA12878/300X/workspaces/Pre-fasta2ubam/work \
    --readgroups_dir /czlab/xuwen/results/NA12878/300X/readGroupList \
    --outdir  /czlab/xuwen/results/NA12878/300X/unmapped_bams \
    -with-report fasta2ubam.html \
    -with-dag fasta2ubam.flowchart.png \
    -with-trace trace.txt \
    -N sherwinmabos@gmail.com \
    -bg &
