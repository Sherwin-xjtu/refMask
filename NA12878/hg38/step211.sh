samtools view -h /czlab/xuwen/DFCI/project2/NA12878/analysis/t2t/NA12878.recal.merged.new.t2t.sorted.bam | grep --color=auto -F -f unique_read_nameschrX.txt - > outchrXh.bam
samtools view -@ 16 -Sb outchrXh.bam > outchrXh.b.bam
samtools index -@ 16 outchrXh.b.bam
