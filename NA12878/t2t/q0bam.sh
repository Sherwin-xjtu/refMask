samtools view -h -@ 16 NA12878.recal.merged.new.t2t.sorted.bam | awk '$5 == 0' > NA12878.recal.merged.new.t2t.sorted.q0.bam
