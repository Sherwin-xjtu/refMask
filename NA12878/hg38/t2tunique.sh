samtools view -h -@ 16 out.b.bam | awk '$5 != 0' > out.b.t2tunique.bam
samtools view -@ 16 -Sb out.b.t2tunique.bam > out.b.t2tunique.b.bam
samtools index -@ 16 out.b.t2tunique.b.bam
