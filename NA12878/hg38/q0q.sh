samtools view -@ 16 -h NA12878.recal.merged.new.sorted.q0.b.chrX.bam | grep --color=auto -F -f unique_q0q0read_nameschrX.txt - > q0qchrXh.bam
cat oheader.txt q0qchrXh.bam > temporary_file && mv temporary_file q0qchrXhh.bam
samtools view -@ 16 -Sb q0qchrXhh.bam > q0qchrXhh.b.bam
samtools index -@ 16 q0qchrXhh.b.bam
samtools depth q0qchrXhh.b.bam > unique_q0q0region_counts.txt
