java -jar /czlab/xuwen/software/JVARKIT/jvarkit.jar samgrep -f unique_read_nameschrX.txt -- /czlab/xuwen/DFCI/project2/NA12878/analysis/t2t/NA12878.recal.merged.new.t2t.sorted.bam > outchrX.bam
samtools view -@ 16 -Sb outchrX.bam > outchrX.b.bam
samtools index -@ 16 outchrX.b.bam
