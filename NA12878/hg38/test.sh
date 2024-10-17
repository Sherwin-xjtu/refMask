cat oheader.txt NA12878.recal.merged.new.sorted.q0.bam > NA12878.recal.merged.new.sorted.q0.h.bam
samtools view -@ 16 -Sb NA12878.recal.merged.new.sorted.q0.h.bam > NA12878.recal.merged.new.sorted.q0.b.bam
samtools index -@ 16 NA12878.recal.merged.new.sorted.q0.b.bam
samtools view -@ 16 -F 4 NA12878.recal.merged.new.sorted.q0.b.bam | cut -f 1 > read_names.txt
java -jar /czlab/xuwen/software/JVARKIT/jvarkit.jar samgrep -f read_names.txt -- /czlab/xuwen/DFCI/project2/NA12878/analysis/t2t/NA12878.recal.merged.new.t2t.sorted.bam > out.bam
samtools view -@ 16 -Sb out.bam > out.b.bam
samtools index -@ 16 out.b.bam
