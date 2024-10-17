samtools view -h -@ 16 NA12878.recal.merged.new.sorted.bam | awk '$5 == 0' > NA12878.recal.merged.new.sorted.q0.bam
