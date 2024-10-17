samtools view -@ 16 -F 4 NA12878.recal.merged.new.sorted.q0.b.chrX.bam | cut -f 1 > read_nameschrX.txt
java -jar /czlab/xuwen/software/JVARKIT/jvarkit.jar samgrep -f read_nameschrX.txt -- /czlab/xuwen/DFCI/project2/NA12878/analysis/t2t/NA12878.recal.merged.new.t2t.sorted.bam > outchrX.bam
samtools view -@ 16 -Sb outchrX.bam > outchrX.b.bam
samtools index -@ 16 outchrX.b.bam