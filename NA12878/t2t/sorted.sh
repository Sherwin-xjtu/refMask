nohup srun --cpus-per-task=16 --mem-per-cpu=16G samtools sort -@ 16 NA12878.recal.merged.new.t2t.bam > NA12878.recal.merged.new.t2t.sorted.bam 2> NA12878.recal.merged.new.t2t.sorted.log &
